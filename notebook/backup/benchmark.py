
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import glob
import json
import time
import argparse
import datetime

import numpy as np
from PIL import Image

import webdataset as wds
import s3torchconnector as s3pt

import torch
import torch.nn as nn
import torchdata
from torchvision.transforms import v2 as tvt

import ray.train
import ray.train.torch


################## BENCHMARK PARAMETERS DEFINITION ###################

def parse_args():
    
    def none_or_int(value):
        if str(value).upper() == 'NONE':
            return None
        return int(value)
    
    def none_or_str(value):
        if str(value).upper() == 'NONE':
            return None
        return str(value)
    
    def str_bool(value):
        if str(value).upper() == 'TRUE':
            return True
        elif str(value).upper() == 'FALSE':
            return False
        else:
            raise TypeError("Must be True or False.")
    
    parser = argparse.ArgumentParser()

    ### Parameters that define dataloader config
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataloader_workers', type=int, default=0)
    parser.add_argument('--dataloader_use_s3pt', type=str_bool, default=False)
    parser.add_argument('--prefetch_size', type=none_or_int, default=2)
    parser.add_argument('--input_dim', type=int, default=224)
    parser.add_argument('--pin_memory', type=str_bool, default=True)

    ### Parameters that define dataset config
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_format', type=str)
    parser.add_argument('--dataset_num_samples', type=int, default=100_000)
    parser.add_argument('--dataset_region', type=none_or_str, default=os.getenv('AWS_REGION'))

    ### Parameters that define model parameters
    parser.add_argument('--model_compute_time', type=none_or_int, default=None) # in miliseconds
    parser.add_argument('--model_num_parameters', type=int, default=1) # in millions of parameters
    
    ### Parameters that define benchmark infrastructure
    parser.add_argument('--ray_workers', type=int, default=2)
    parser.add_argument('--ray_cpus_per_worker', type=int, default=8)
    parser.add_argument('--ray_use_gpu', type=str_bool, default=False)
   
    ### Parameters that define checkpointing config
    parser.add_argument('--ckpt_steps', type=int, default=0)
    parser.add_argument('--ckpt_mode', type=str, default='disk')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/')
    parser.add_argument('--ckpt_region', type=none_or_str, default=os.getenv('AWS_REGION'))

    ### Some other parameters for logging results
    parser.add_argument('--log_directory', type=none_or_str, default=os.path.join(os.getenv('EKS_MOUNTPOINT_DIR', '.'), 'logs'))
    parser.add_argument('--benchmark_name', type=none_or_str, default=f'benchmark-{datetime.datetime.now().strftime("%Y%m%d%H%M%S-%f")}')

    return parser.parse_known_args()


################## MODEL IMPLEMENTATION ###################

class ModelMock(torch.nn.Module):
    '''Model mock to emulate a computation of a training step'''
    def __init__(self, config):
        super().__init__()
        self.model = torch.nn.Linear(config.model_num_parameters * 1_000_000, 1)
        self.config = config
    
    def forward(self, data, target, epoch, step):
        if self.config.model_compute_time > 0:
            return time.sleep(self.config.model_compute_time / 1_000)

        if (
            ray.train.get_context().get_world_rank() == 0 and
            self.config.ckpt_steps > 0 and
            step % self.config.ckpt_steps == 0
        ):
            return self.save_checkpoint(epoch, step)

    def save_checkpoint(self, epoch, step):
        if self.config.ckpt_mode == 's3pt':
            return save_checkpoint_s3pt(self.model, self.config.ckpt_region, self.config.ckpt_path, epoch, step)
        elif self.config.ckpt_mode == 'disk':
            return save_checkpoint_disk(self.model, self.config.ckpt_path, epoch, step)
        else:
            raise NotImplementedError("Unknown checkpoint mode '%s'.." % self.config.ckpt_mode)


def save_checkpoint_s3pt(model, region, uri, epoch_id, step_id):
    path = os.path.join(uri, f"epoch-{epoch_id}-step-{step_id}.ckpt")
    checkpoint = s3pt.S3Checkpoint(region=region)
    start_time = time.perf_counter()
    with checkpoint.writer(path) as writer:
        torch.save(model.state_dict(), writer)
    end_time = time.perf_counter()
    save_time = end_time - start_time
    print_from_rank(f"Saving checkpoint to {uri} took {save_time} seconds..")
    return save_time

def save_checkpoint_disk(model, uri, epoch_id, step_id):
    if not os.path.exists(uri):
        os.makedirs(uri)
    path = os.path.join(uri, f"epoch-{epoch_id}-step-{step_id}.ckpt")
    start_time = time.perf_counter()
    torch.save(model.state_dict(), path)
    end_time = time.perf_counter()
    save_time = end_time - start_time
    print_from_rank(f"Saving checkpoint to {path} took {save_time} seconds..")
    return save_time
            

################## DATASET IMPLEMENTATIONS ###################

class MapDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform):
        self._files = np.array(files)
        self._transform = transform
   
    @staticmethod
    def _get_label(file):
        return file.split(os.path.sep)[-2]
    
    @staticmethod
    def _read(file):
        return Image.open(file).convert('RGB')
    
    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, idx):
        file = self._files[idx]
        sample = self._transform(self._read(file))
        label = int(self._get_label(file))    # Labels in [0, MAX) range
        return sample, label

def _make_pt_dataset(config, transform):
    # Create a dataset from individual image files
    
    files = glob.glob(config.dataset_path + '/**/*.jpg')
    dataset = MapDataset(files, transform)
    return dataset

def _make_wds_dataset(config, transform):
    # Create a WebDataset from tar files and apply transformations
    
    def _create_sample(sample):
        label, img = sample['__key__'], sample['jpg']
        img = transform(img)
        label = int(label.split('/')[-2])
        return img, label
    
    files = glob.glob(config.dataset_path + '/*.tar')        
    dataset = wds.WebDataset(files, shardshuffle=True, resampled=True, nodesplitter=wds.split_by_node)
    dataset = dataset.decode('pil')
    dataset = dataset.map(_create_sample)
    dataset = dataset.with_epoch(config.dataset_num_samples // (config.ray_workers * config.dataloader_workers))
    return dataset

def _make_s3pt_dataset(config, transform):

    def _tar_to_tuple(s3object):
        return s3object.key, torchdata.datapipes.utils.StreamWrapper(s3object)
    
    def _create_sample(item):
        label, img = item
        img = transform(Image.open(img).convert('RGB'))
        label = int(label.split('/')[-2])
        return img, label

    dataset = s3pt.S3IterableDataset.from_prefix(config.dataset_path, region=config.dataset_region)
    dataset = torchdata.datapipes.iter.IterableWrapper(dataset)
    if config.dataloader_workers > 0:
        dataset = dataset.sharding_filter()
    dataset = dataset.map(_tar_to_tuple)
    dataset = dataset.load_from_tar()
    dataset = dataset.map(_create_sample)
    return dataset


################## BENCHMARK IMPLEMENTATIONS #################
def build_dataloader(config):
    # Define image transformations and build the dataloader based on dataset format
    transform = tvt.Compose([
        tvt.ToImage(),
        tvt.ToDtype(torch.uint8, scale=True),
        tvt.RandomResizedCrop(size=(config.input_dim, config.input_dim), antialias=False), #antialias=True
        tvt.ToDtype(torch.float32, scale=True),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Build dataset
    if config.dataset_format == 'jpg':
        dataset = _make_pt_dataset(config, transform)
    elif config.dataset_format == 'tar':
        if config.dataloader_use_s3pt:
            dataset = _make_s3pt_dataset(config, transform)
        else:
            dataset = _make_wds_dataset(config, transform)
    else:
        raise NotImplementedError("Unknown dataset format '%s'.." % config.dataset_format)


    return torch.utils.data.DataLoader(
        dataset,
        num_workers=config.dataloader_workers,
        batch_size=config.batch_size,
        prefetch_factor=config.prefetch_size,
        pin_memory=config.pin_memory
    )


def build_model(config):
    # Build a model or a model mock based on provided config
    if config.model_compute_time is not None:
        model = ModelMock(config)
    else:
        raise NotImplementedError("Need to set compute time explicitely..")
    return model


def train_model(model, dataloader, config):
    # Train model and collect metrics
    metrics = {}
    img_tot_list, ep_times, ckpt_times = [], [], []
    t_train_start = t_epoch_start = time.perf_counter()

    for epoch in range(config.epochs):
        img_tot = 0
        
        for step, (images, labels) in enumerate(dataloader, 1):

            # Perform a training step and optionally save checkpoint
            batch_size = len(images)
            img_tot += batch_size

            result = model(images, labels, epoch, step)
            
            if result:
                ckpt_times.append(result)

            if step % 50 == 0:
                print_from_rank(f"Epoch = {epoch} | Step = {step}")

        # Record metrics for each epoch
        img_tot_list.append(img_tot)
        ep_times.append(time.perf_counter() - t_epoch_start)
        t_epoch_start = time.perf_counter()

    # Summarize training metrics
    t_train_tot = time.perf_counter() - t_train_start
    metrics['training_time'] = t_train_tot
    metrics['samples_per_second'] = sum(img_tot_list) / t_train_tot
    metrics['samples_processed_total'] = sum(img_tot_list)
    metrics.update({f't_epoch_{i}': t for i, t in enumerate(ep_times, 1)})
    metrics.update({f't_ckpt_{i}': t for i, t in enumerate(ckpt_times, 1)})
    if ckpt_times:
        metrics['t_ckpt_ave'] = sum(ckpt_times) / len(ckpt_times)
    return metrics


################## HELPER FUNCTIONS #################
def print_from_rank(msg, rank=0):
    if ray.train.get_context().get_world_rank() == rank:
        print(f'[r:{rank}]:', msg)


############ MAIN EXECUTABLE FUNCTION ##############
def main_fn(config):

    # Print debugging information and configuration
    print_from_rank("Benchmarking params:\n" + json.dumps(vars(config), indent=2))
    print_from_rank("Environment variables:\n")
    for k, v in os.environ.items():
        print_from_rank(f'{k}={v}')

    # # Print example dataset files for debugging
    filelist_gen = glob.iglob(os.path.join(config.dataset_path, '**', '*'), recursive=True)
    print_from_rank(f"Files in {config.dataset_path}:")
    for i, f in enumerate(filelist_gen):
        print_from_rank(" - " + f)
        if i > 10: break
    
    # Step #1: Build dataloader and prepare it for Ray distributed environment
    dataloader = build_dataloader(config)
    dataloader = ray.train.torch.prepare_data_loader(dataloader)

    # Step #2: Build model and prepare it for Ray distributed environment
    model = build_model(config)
    model = ray.train.torch.prepare_model(model)

    # Step #3: Train the model and collect metrics
    metrics = train_model(model, dataloader, config)
    
    # Step #4: Log metrics and save to S3      
    os.makedirs(config.log_directory, exist_ok=True)
    log_file = os.path.join(config.log_directory, config.benchmark_name + '.json')
    with open(log_file, 'w') as f:
        json.dump(metrics, f)

    print_from_rank(f"Logged the following metrics to '{log_file}':\n" + json.dumps(metrics, indent=2))

    time.sleep(3)

    return
        

################## ENTRY POINT #################
if __name__ == '__main__':

    # Parse configuration arguments
    train_config, _ = parse_args()

    # Set up scaling configuration for Ray Trainer
    scaling_config = ray.train.ScalingConfig(
        num_workers=train_config.ray_workers,
        use_gpu=train_config.ray_use_gpu,
        resources_per_worker={
            'CPU': train_config.ray_cpus_per_worker
        })
    
    # Initialize Ray TorchTrainer with main function
    trainer = ray.train.torch.TorchTrainer(
        main_fn,
        scaling_config=scaling_config,
        train_loop_config=train_config)

    # Run the distributed training job
    result = trainer.fit()
