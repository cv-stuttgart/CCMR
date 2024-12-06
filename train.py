from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from CCMR import CCMR
import evaluate
import datasets

import logging
from config.config_loader import load_json_config
import json
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    print("NO GRADSCALING! EXITING.....")
    exit(0)
   

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 50
VAL_FREQ = 2000


def sequence_loss(flow_preds, flow_gt, valid, loss_type, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    eps = 0.01  # for robust loss terms

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()

        num_valid_pixels = torch.sum(valid[:, None])
        num_all_pixels = torch.sum(torch.ones_like(valid, dtype=float))
        valid_ratio = num_valid_pixels / num_all_pixels

        if loss_type == "L2":
            l2_norm = torch.sum((valid[:, None] * i_loss)**2 + 0.000001, dim=1).sqrt()
            l2_loss = l2_norm.mean() / valid_ratio
            flow_loss += i_weight * l2_loss

        elif loss_type == "rob_samplewise07":
            def reciprocal_no_nan(x, condition):
                y = torch.zeros_like(x)
                y[condition] = torch.reciprocal(x[condition])
                return y

            num_valid_pixels = torch.sum(valid, dim=[1, 2])
            num_all_pixels = torch.sum(torch.ones_like(valid, dtype=float), dim=[1, 2])
            valid_ratio = valid.float().mean(dim=[1, 2])  # num_valid_pixels / num_all_pixels

            l2_norm = torch.sum((valid[:, None] * i_loss)**2 + 0.000001, dim=1).sqrt()
            norm_imgwise = l2_norm.mean(dim=[1, 2])
            validity = valid_ratio > 0.1
            rob_norm_imagewise = pow(norm_imgwise + eps, 0.7) * reciprocal_no_nan(valid_ratio, validity)
            flow_loss += i_weight * rob_norm_imagewise.sum() / validity.sum()

        else:
            raise ValueError(f'No loss with name "{loss_type}" is implemented.')

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        "flow_loss": flow_loss,
        'epe': epe.mean().item()
    }

    return metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(phase, model, local_step):
    """ Create the optimizer and learning rate scheduler """

    optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"][phase],
                            weight_decay=config["train"]["wdecay"][phase], eps=config["epsilon"])

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, config["train"]["lr"][phase], config["train"]["num_steps"][phase]+100,
        pct_start=config["lr_peak"], cycle_momentum=False, anneal_strategy='linear')

    # For the case the training is resumed within one phase: To continue from the last checkpoint's iteration learning rate.
    for i in range(local_step + 1):
        scheduler.step()
    return optimizer, scheduler


class StatsLogger:
    def __init__(self, name, current_steps, phase):
        self.total_steps = current_steps
        self.phase = phase
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=os.path.join("checkpoints", name))
        self.metrics_file = os.path.join("checkpoints", name, "lrs.csv")
        self.time = datetime.now()
        self.logger = logging.getLogger("ccmr.stats")

        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as file:
                file.write("step,lr\n")

    def set_phase(self, phase, dataset):
        self.phase = phase
        self.dataset_being_trained = dataset

    def _print_training_status(self, lr):
        now = datetime.now()
        time_diff = now - self.time
        self.time = now

        training_str = "[number of steps: {0:6d}, lr: {1:2.7f}, dataset: {2}, phase: {3}, duration: {4:4.2f}, time:{5}] "
        training_str = training_str.format(self.total_steps+1, lr, self.dataset_being_trained,
                                           self.phase, time_diff.total_seconds(), now)

        metrics_str = ",".join(f"{key}:{(value/SUM_FREQ):8.4f} "for key, value in self.running_loss.items())

        self.logger.info("%s %s", metrics_str, training_str)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)

    def push(self, metrics, lr):
        self.total_steps += 1  # assume local step starts from -1, as it actually does.

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        with open(self.metrics_file, "a") as file:
            file.write("{:6d},{:10.7f}\n".format(self.total_steps, lr))

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status(lr)
            self.running_loss = {}

    def write_dict(self, results):

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def save_model_and_checkpoint(model, steps, phase, saving_policy="limited"):
    if saving_policy == "unlimited":
        checkpoint_values_path = 'checkpoints/%s/_%s_phase%d_%d.pth' % (config["name"], config["train"]["dataset"][phase], phase, steps)
        torch.save(model.state_dict(), checkpoint_values_path)
    elif saving_policy == "limited":
        checkpoint_values_path = 'checkpoints/%s/%s_phase%d_%d.pth' % (config["name"], config["train"]["dataset"][phase], phase, steps)
        torch.save(model.state_dict(), checkpoint_values_path)
        checkpoint_txt_path = 'checkpoints/%s/checkpoint.txt' % config["name"]
        create_checkpoint_file(checkpoint_txt_path, phase, steps, checkpoint_values_path, config)
    else:
        assert ValueError("Unknown saving policy given.")


def create_checkpoint_file(txtfile_path, phase, current_steps, checkpoint_name, config):
    if not os.path.exists(txtfile_path):
        with open(txtfile_path, 'w') as file:
            dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": None}
            json.dump(dict, file)

    else:
        with open(txtfile_path) as file:
            checkpoint_config = json.load(file)

        with open(txtfile_path, "w") as file:
            if checkpoint_config["newer"] is None:
                dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": None}
                json.dump(dict, file)
            elif (checkpoint_config["newer"] is not None) and (checkpoint_config["older"] is None):
                dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": checkpoint_config["newer"]}
                json.dump(dict, file)
            else:
                dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": checkpoint_config["newer"]}
                json.dump(dict, file)
                # remove the older file:
                name = config["name"]
                older_file_path = checkpoint_config["older"]
                file_path_to_be_removed = older_file_path
                if os.path.exists(file_path_to_be_removed):
                    os.remove(file_path_to_be_removed)
                else:
                    logger = logging.getLogger("ccmr.saving")
                    logger.error("Checkpoint file did not exist. old checkpoint.txt: %s, new checkpoint.txt: %s", str(checkpoint_config), str(dict))


def fetch_model(phase):
    model = nn.DataParallel(CCMR(config), device_ids=config["gpus"])

    print("Parameter Count: %d" % count_parameters(model))

    model.cuda()
    model.train()

    if config["train"]["dataset"][phase] != 'chairs':
        model.module.freeze_bn()
    return model


def fetch_data(phase):
    data_loader, _ = datasets.fetch_dataloader(config, phase)
    while True:
        for data_blob in data_loader:
            yield [x.cuda() for x in data_blob]


def passed_steps(phase):
    steps = 0
    if phase != 0:
        steps = sum(config["train"]["num_steps"][:phase])

    return steps


def training_step(model, data_group, optimizer, phase, scaler):
    iterations = config["train"]["iters"]
    for data in data_group:
        image1, image2, flow, valid = data
        if config["add_noise"]:
            stdv = np.random.uniform(0.0, 5.0)
            image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
        flow_predictions = model(image1, image2, iters=iterations)

        metrics = sequence_loss(flow_predictions, flow, valid, config["train"]["loss"][phase], config["train"]["gamma"][phase])

        if torch.isnan(metrics["flow_loss"]):
            logger = logging.getLogger("ccmr.saving")
            logger.error("nan loss during training. Exiting...")
            exit(0)

        scaler.scale(metrics["flow_loss"]/config["grad_acc"][phase]).backward()  # scale based on the gradient accumulation step.

    # after processing the "complete batch" --> update parameters:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
    scaler.step(optimizer)
    scaler.update()
    return metrics


def train_single_phase(model, optimizer, scheduler, init_local_step, num_steps, stats_logger, phase):
    data_generator = fetch_data(phase)
    eval_iters_list = config["train"]["eval_iters"][phase]
    scaler = GradScaler(enabled=config["mixed_precision"])

    for local_step in range(init_local_step, num_steps - 1):
        data_group = []
        optimizer.zero_grad()
        for acc_step in range(config["grad_acc"][phase]):
            data = next(data_generator)
            data_group.append(data)
            metrics = training_step(model, data_group, optimizer, phase, scaler)

        scheduler.step()

        stats_logger.push(metrics, scheduler.get_last_lr()[0])

        if (local_step + 1) % VAL_FREQ == VAL_FREQ - 1:  # save checkpoint now
            save_model_and_checkpoint(model, local_step + 2, phase, "limited")
            results = {}
            if config["train"]["validation"][phase] == 'chairs':
                results.update(evaluate.validate_chairs(model.module, iters=eval_iters_list))
            elif config["train"]["validation"][phase] == 'sintel':
                results.update(evaluate.validate_sintel_during_training(model.module, iters = eval_iters_list, warm=True))
            elif config["train"]["validation"][phase] == 'kitti':
                results.update(evaluate.validate_kitti(model.module, iters=eval_iters_list))
            elif config["train"]["validation"][phase] == 'kitti_split':
                results.update(evaluate.validate_kitti_split(model.module, iters=eval_iters_list))

            stats_logger.write_dict(results)
            model.train()

            if config["train"]["dataset"][phase] != 'chairs':
                model.module.freeze_bn()


def train_phases():
    num_phases = len(config["train"]["num_steps"])
    init_phase = config["current_phase"]
    local_step = config["current_steps"]  # local step is the current step in the current phase.
    passed_train_steps = passed_steps(init_phase)
    stats_logger = StatsLogger(config["name"], local_step + passed_train_steps, init_phase)
    datasets = config["train"]["dataset"]
    num_steps = config["train"]["num_steps"]

    if config["train"]["restore_ckpt"] is not None:
        state_dict = torch.load(config["train"]["restore_ckpt"])
        print("Loading checkpoint from %s....." % config["train"]["restore_ckpt"])
    else:
        state_dict = None

    for phase in range(init_phase, num_phases):
        torch.manual_seed(1234)
        np.random.seed(1234)
        stats_logger.set_phase(phase, datasets[phase])
        model = fetch_model(phase)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        optimizer, scheduler = fetch_optimizer(phase, model, local_step)

        train_single_phase(model, optimizer, scheduler, local_step, num_steps[phase], stats_logger, phase)
        save_model_and_checkpoint(model, num_steps[phase], phase, "unlimited")

        local_step = -1
        state_dict = model.state_dict()

    stats_logger.close()


def train(config):
    logger = logging.getLogger("ccmr.train")

    if config["train"]["restore_ckpt"] is None:
        possible_checkpoint_file = os.path.join("checkpoints", config["name"], "checkpoint.txt")
        if (os.path.exists(possible_checkpoint_file)):
            file = open(possible_checkpoint_file)
            checkpoint_configs = json.load(file)
            config["current_phase"] = checkpoint_configs["phase"]
            config["train"]["restore_ckpt"] = checkpoint_configs["newer"]
            config["current_steps"] = checkpoint_configs["current_steps"] - 1  # local step is the index of the steps.

    logger.info(config)
    train_phases()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to the configuration file')
    args = parser.parse_args()

    config = load_json_config(args.config)
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1000"  # uncomment in case of memory fragmentation. Note: small split size might slow down the training!
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir(f'checkpoints/{config["name"]}'):
        os.mkdir(f'checkpoints/{config["name"]}')

    filehandler = logging.FileHandler(f"checkpoints/{config['name']}/log.txt")
    # In the file, write Info or the other things with higer level than info: error, warning and stuff.
    filehandler.setLevel(logging.INFO)

    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)

    logger = logging.getLogger("ccmr")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info("starting to train")

    train(config)