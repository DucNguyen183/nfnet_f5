import argparse
import math
import PIL
import time
import yaml
from pathlib import Path
from PIL.Image import Image
import copy
import datetime
import matplotlib.pyplot as plt
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop

from dataset import get_dataset
from nfnets import NFNet, SGD_AGC, pretrained_nfnet
import os


def train(config:dict) -> None:
    model_path = 'model/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if config['device'].startswith('cuda'):
        if torch.cuda.is_available():
            print(f"Using CUDA{torch.version.cuda} with cuDNN{torch.backends.cudnn.version()}")
        else:
            raise ValueError("You specified to use cuda device, but cuda is not available.")
    
    if config['pretrained'] is not None:
        model = pretrained_nfnet(
            path=config['pretrained'], 
            stochdepth_rate=config['stochdepth_rate'],
            alpha=config['alpha'],
            activation=config['activation']
            )
    else:
        model = NFNet(
            num_classes=config['num_classes'], 
            variant=config['variant'], 
            stochdepth_rate=config['stochdepth_rate'], 
            alpha=config['alpha'],
            se_ratio=config['se_ratio'],
            activation=config['activation']
            )

    device = config['device']

    dataloader = get_dataset(config['dataset'], model.train_imsize, config['batch_size'])
    if config['scale_lr']:
        learning_rate = config['learning_rate']*config['batch_size']/256
    else:
        learning_rate = config['learning_rate']

    if not config['do_clip']:
        config['clipping'] = None

    if config['use_fp16']:
        model.half()

    model.to(device) # "memory_format=torch.channels_last" TBD

    optimizer = SGD_AGC(
        # The optimizer needs all parameter names 
        # to filter them by hand later
        named_params=model.named_parameters(), 
        lr=learning_rate,
        momentum=config['momentum'],
        clipping=config['clipping'],
        weight_decay=config['weight_decay'], 
        nesterov=config['nesterov']
        )
    
    # Find desired parameters and exclude them 
    # from weight decay and clipping
    for group in optimizer.param_groups:
        name = group['name']
        
        if model.exclude_from_weight_decay(name):
            group['weight_decay'] = 0

        if model.exclude_from_clipping(name):
            group['clipping'] = None

    criterion = nn.CrossEntropyLoss()

    runs_dir = Path('runs')
    run_index = 0
    while (runs_dir / ('run' + str(run_index))).exists():
        run_index += 1
    runs_dir = runs_dir / ('run' + str(run_index))
    runs_dir.mkdir(exist_ok=False, parents=True)
    checkpoints_dir = runs_dir / 'checkpoints'
    checkpoints_dir.mkdir()

    writer = SummaryWriter(str(runs_dir))
    scaler = amp.GradScaler()
    tic = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(config['epochs']):
        print("{} epoch {}:".format(datetime.datetime.now(), epoch))
        for phase in ['train', 'val']:
            running_loss = 0.0
            correct_labels = 0
            epoch_time = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for data, target in dataloader[phase]:
                inputs = data.half().to(device) if config['use_fp16'] else data.to(device)
                targets = target.to(device)

                optimizer.zero_grad()

                with amp.autocast(enabled=config['amp']):
                    output = model(inputs)
                loss = criterion(output, targets)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()*inputs.size(0)
                _, predicted = torch.max(output, 1)
                correct_labels += (predicted == targets).sum().item()

            epoch_loss = running_loss/len(dataloader[phase].dataset)
            epoch_accuracy = 100. * correct_labels / len(dataloader[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))
            if phase == 'val' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_path + 'id_classify_nfnet_F5.pt')
        print("=============================================")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train NFNets.')
    # parser.add_argument('--config', type=Path, help='Path to config.yaml', default='default_config.yaml')
    parser.add_argument('--batch-size', type=int, help='Training batch size', default=None)
    parser.add_argument('--overfit', const=True, default=False, nargs='?', help='Crop the dataset to the batch size and force model to (hopefully) overfit')
    parser.add_argument('--variant', type=str, help='NFNet variant to train', default=None)
    # parser.add_argument('--pretrained', type=Path, help='Path to pre-trained weights in haiku format', default=None)
    args = parser.parse_args()
    config = 'default_config.yaml'
    if not args.config.exists():
        print(f"Config file \"{config}\" does not exist!\n")
        exit()

    with args.config.open() as file:
        config = yaml.safe_load(file)

    # Override config.yaml settings with command line settings
    for arg in vars(args):
        if getattr(args, arg) is not None and arg in config:
            config[arg] = getattr(args, arg)
    pretrained = 'pretrained/F5_haiku.npz'
    config['pretrained'] = pretrained

    train(config=config)
