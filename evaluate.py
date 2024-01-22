import sys
sys.path.append('core')
import argparse
import os
import os.path as osp
import numpy as np
import torch
import datasets
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
from CCMR import CCMR
from config.config_loader import cpy_eval_args_to_config
from custom_logger import init_logger
import logging
from tqdm import tqdm

@torch.no_grad()
def create_kitti_submission(model, iters, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)
    coarsest_scale = 16
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in tqdm(range(len(test_dataset))):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti', coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1=image1, image2=image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_sintel_submission(model, iters, output_path='sintel_submission', split= "test"):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    coarsest_scale = 16
   
    for dstype in ['final', 'clean']:
        test_dataset = datasets.MpiSintel(split=split, aug_params=None, dstype=dstype, show_extra_info=True)

        flow_prev, sequence_prev = None, None
        for test_id in tqdm(range(len(test_dataset))):
            if split == "test":
                image1, image2, (sequence, frame) = test_dataset[test_id]
            elif split == "training":
                image1, image2,_,_, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            _, flow_pr = model(image1=image1, image2=image2, iters=iters, flow_init=flow_prev, test_mode=True)#cold warm
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
            
            init_flow_for_next_low, _ =  model(image1=image1, image2=image2, iters=iters, flow_init=None, test_mode=True)#cold warm
            flow_prev = forward_interpolate(init_flow_for_next_low[0])[None].cuda()#cold warm
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)

            sequence_prev = sequence

@torch.no_grad()
def validate_sintel(model, iters, model_path):
    """ Evaluate trained model on Sintel(train) clean + final passes on train split """
    logger = logging.getLogger('eval.sintel')
    logger_for_eval_all = logging.getLogger('eval_all.sintel')
    model.eval()
    results = {}
    coarsest_scale = 16
   
    epe_sequence={}
    
    for dstype in ['clean', 'final']:

        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, show_extra_info = True)
        epe_list = []
        efel_list = []
        flow_prev, sequence_prev = None, None
        sequences = []


        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _, (sequence, frame) = val_dataset[val_id]
            frame += 1

            if sequence not in sequences:
                sequences.append(sequence)

            if f"{sequence}_epe" not in epe_sequence:
                epe_sequence[f"{sequence}_epe"] = 0
                epe_sequence[f"{sequence}_count"] = 0

            padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            if sequence != sequence_prev:
                flow_prev = None
            
            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()
            flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            #*****
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            epe = epe.view(-1)
            mag = mag.view(-1)
            efel = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            efel_list.append(efel.cpu().numpy())
            #******

            #logger.info("%s %s %d  EPE: %f, " % (dstype, sequence, frame, epe.mean()))
            sequence_prev = sequence

            epe_sequence[f"{sequence}_epe"] += epe.mean()
            epe_sequence[f"{sequence}_count"] += 1

        for seq in sequences:
            epe_per_seq = epe_sequence[f"{seq}_epe"]/epe_sequence[f"{seq}_count"]
            logger.info("Sintel validation, iters: %d, type: %s  seq:(%s)  EPE: %f" % (sum([8, 10, 10, 10]), dstype, seq, epe_per_seq)) 
                
        efel_list = np.concatenate(efel_list)
        FL = 100 * np.mean(efel_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        logger.info("Sintel validation iters:%d (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, FL-Error: %f:" % (sum([8, 10, 10, 10]), dstype, epe, px1, px3, px5, FL))
        logger_for_eval_all.info("Sintel iters:%d validation of the model: %s , : (%s) EPE: %f , FL-Error: %f:" % (sum([8, 10, 10, 10]), model_path, dstype, epe, FL))
        results[dstype] = np.mean(epe_list)


    return results

@torch.no_grad()
def validate_kitti(model, iters, padding="kitti"):
    """ Peform validation using the KITTI-2015 (train) split """
    logger = logging.getLogger("eval.kitti")
    model.eval()
    val_dataset = datasets.KITTI(split='training')
    out_list, epe_list = [], []
    coarsest_scale = 16
    for data_id in tqdm(range(len(val_dataset))): # uses len() and updates correctly

        image1, image2, flow_gt, valid_gt = val_dataset[data_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, mode=padding, coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1, image2)
        
        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    logger.warning("Kitti validation:%f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1", "True")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--model_type', choices=['CCMR', 'CCMR+'], default="CCMR+", help="Choose \"CCMR\" or \"CCMR+\" for the 3-scale and 4-scale version, respectively.")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--num_scales', type=int, help="3 for CCMR and 4 for CCMR+", default=4)
    parser.add_argument('--mixed_precision', help='use mixed precision', type = str2bool, default = True)

    args = parser.parse_args()
    config = cpy_eval_args_to_config(args)
    
    
    if config["model_type"] == "CCMR":
        sintel_iters = [8, 10, 15]
        kitti_iters = [6, 8, 30] 
        config["num_scales"] = 3
        
    elif config["model_type"] == "CCMR+":
        sintel_iters = [8, 10, 10, 10]
        kitti_iters = [35, 35, 5, 15]
        config["num_scales"] = 4
    
    model = torch.nn.DataParallel(CCMR(config))
    model.load_state_dict(torch.load(config["model"]))

    path = os.path.join(os.path.dirname((config["model"])), "eval.log")
    logger = init_logger("eval", path, stream_level= logging.WARNING)
    logger = init_logger("eval_all", "./eval_all.log")
    
    model.cuda()
    model.eval()

    with torch.no_grad():

        if config["dataset"] == 'sintel':
            validate_sintel(model=model.module, iters= sintel_iters, model_path=config["model"])

        elif config["dataset"] == 'sintel_test':
            out_path = osp.join(os.path.dirname(config["model"]), 'sintel_test')
            create_sintel_submission(model, iters=sintel_iters, output_path=out_path)

        elif config["dataset"] == 'kitti':
            validate_kitti(model.module, iters=kitti_iters)
    
        elif config["dataset"] == "kitti_test":
            create_kitti_submission(model, iter=kitti_iters, output_path=osp.join(os.path.dirname(config["model"]), 'kitti_test', 'flow'))

       