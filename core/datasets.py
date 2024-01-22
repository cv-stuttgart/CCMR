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

