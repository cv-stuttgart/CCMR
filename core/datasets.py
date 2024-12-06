# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import numpy as np
import torch
import torch.utils.data as data
import os
import random
from glob import glob
import os.path as osp
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
import parse
import logging
extention = './data/'

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, show_extra_info=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.show_extra_info = show_extra_info
        self.max_flow = None

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_img_generic(self.image_list[index][0])
            img2 = frame_utils.read_img_generic(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        if self.flow_list[index] is not None:
            flow, valid = frame_utils.read_flow_generic(self.flow_list[index])
        else:
            flow, valid = None, None

        img1 = frame_utils.read_img_generic(self.image_list[index][0])
        img2 = frame_utils.read_img_generic(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if flow is not None:
            flow = np.array(flow).astype(np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                assert np.all(valid), "ground-truth flow is sparse, but dense augmentor is used"
                img1, img2, flow = self.augmentor(img1, img2, flow)
                valid = np.ones(img1.shape[:-1], dtype=np.bool)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        if flow is not None:
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is None:
            valid = torch.zeros(*img1.shape[1:], dtype=torch.bool)
        else:
            valid = torch.from_numpy(valid).to(dtype=torch.bool)

        if flow is not None:
            if not self.sparse:
                valid &= (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

            if self.max_flow is not None:
                valid &= (flow[0].abs() < self.max_flow) & (flow[1].abs() < self.max_flow)

            valid &= torch.isfinite(flow[0]) & torch.isfinite(flow[1])
            flow[:, ~(torch.isfinite(flow[0]) & torch.isfinite(flow[1]))] = 0.0

        if self.show_extra_info:
            return img1, img2, flow, valid.float(), self.extra_info[index]
        else:
            return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='sintel', dstype='clean', show_extra_info=False):
        root = extention + root
        super(MpiSintel, self).__init__(aug_params, show_extra_info=show_extra_info)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='kitti15/dataset'):
        root = extention + root
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='fc/data'):
        root = extention + root
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/fth', split='training', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)
        root = extention + root

        if split == 'training':
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')) )
                        flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                        for i in range(len(flows)-1):
                            if direction == 'into_future':
                                self.image_list += [ [images[i], images[i+1]] ]
                                self.flow_list += [ flows[i] ]
                            elif direction == 'into_past':
                                self.image_list += [ [images[i+1], images[i]] ]
                                self.flow_list += [ flows[i+1] ]

        elif split == 'validation':
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TEST/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TEST/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')))
                        flows = sorted(glob(osp.join(fdir, '*.pfm')))
                        for i in range(len(flows) - 1):
                            if direction == 'into_future':
                                self.image_list += [[images[i], images[i + 1]]]
                                self.flow_list += [flows[i]]
                            elif direction == 'into_past':
                                self.image_list += [[images[i + 1], images[i]]]
                                self.flow_list += [flows[i + 1]]

                valid_list = np.loadtxt('things_val_test_set.txt', dtype=np.int32)
                self.image_list = [self.image_list[ind] for ind, sel in enumerate(valid_list) if sel]
                self.flow_list = [self.flow_list[ind] for ind, sel in enumerate(valid_list) if sel]
      

class Viper(FlowDataset):
    def __init__(self, aug_params=None, root='viper', split='train', show_extra_info=False):
        super(Viper, self).__init__(aug_params, show_extra_info=show_extra_info, sparse=True)

        root = osp.join(extention, root, split)
        img_root = osp.join(root, 'img')
        flow_root = osp.join(root, 'flow')

        if split == 'train':            # ignore large outliers
            self.max_flow = 2048

        flows = []
        imgs = []
        info = []

        for group in sorted(os.listdir(img_root)):
            for img1 in sorted(os.listdir(osp.join(img_root, group))):
                num = parse.parse(group + "_{:05d}.jpg", img1)[0]

                img1 = osp.join(img_root, group, img1)
                img2 = osp.join(img_root, group, f"{group}_{num + 1:05d}.jpg")
                flow = osp.join(flow_root, group, f"{group}_{num:05d}.npz")

                if not osp.exists(img2) or not osp.exists(flow):
                    continue 

                if osp.getsize(flow) == 0 or osp.getsize(img1) == 0 or osp.getsize(img2) == 0:
                    continue 

                imgs += [(img1, img2)]
                flows += [flow]
                info += [(group, num)]

        self.image_list = imgs
        self.flow_list = flows
        self.extra_info = info


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='HD1k'):
        root = extention + root
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, phase, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args["train"]["dataset"][phase] == 'chairs':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args["train"]["dataset"][phase] == 'things':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args["train"]["dataset"][phase] == 'sintel':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        kitti = KITTI({'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things
          
    elif args["train"]["dataset"][phase] == 'kitti4_vip6':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}
        kitti = KITTI(aug_params)

        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}
        hd1k = HD1K(aug_params)

        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.8, 'max_scale': 0.1, 'do_flip': True}
        viper = Viper(aug_params)

        train_dataset = 600 * kitti + 10 * hd1k + 13 * viper

    elif args["train"]["dataset"][phase] == 'sintel_f10_c7':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        kitti = KITTI({'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        train_dataset = 70*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

    elif args["train"]["dataset"][phase] == 'sintel_f10_c5':
        aug_params = {'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        kitti = KITTI({'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args["train"]["image_size"][phase], 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        train_dataset = 50*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things
    
    else:
        ds = args["train"]["dataset"][phase]
        raise ValueError(f"unsupported dataset type {ds}")

    train_loader = data.DataLoader(train_dataset, batch_size=args["train"]["batch_size"][phase], 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    logger = logging.getLogger("ccmr.train")
    logger.info('Training with %d image pairs' % len(train_dataset))
    return train_loader, len(train_dataset)

