import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch import optim
from collections import defaultdict
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataset import SateliteDataset, ToTensor2
from unet import UNet
from loss import dice_loss
import time
import copy

batch_size = 8
num_epochs = 300
validation_split = .2

dataset = SateliteDataset('data', transform=ToTensor2())
dataset_size = len(dataset)

indices = list(range(dataset_size))

np.random.seed(42)
np.random.shuffle(indices)
print(indices)

split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
dataloaders['val'] = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(3, 1)
model = model.to(device)

criterion = nn.BCELoss()
def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = criterion(pred, target)
    target = target.unsqueeze(1)
    dice = dice_loss(pred, target)

    loss = bce  * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for data in dataloaders[phase]:
                inputs = data['image'].to(device)
                labels = data['gt'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'model_best.pk')
        time_elapsed = time.time() - since
        # torch.save(model.state_dict(), 'model_epoch' + str(epoch) + '.pk')
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
torch.save(model.state_dict(), 'model_save.pk')