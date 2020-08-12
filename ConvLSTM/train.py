''' Kl - CC '''
''' ConvT -> Conv + Up '''
''' Look for better encoders than S3D '''
''' Output 32 frames with S3D '''
''' Finetuning using lr sched '''
''' Using Decoder to generate 32 gaussians '''
''' Using 2D Decoder for Sliding Window '''
''' Backbone with GRU at highest level of heirarchy '''

import argparse
import glob, os
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.autograd import Variable
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from dataloader import DHF1KDataset
from loss import *
import cv2
from model_lstm import CombinedModel
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs',default=60, type=int)
parser.add_argument('--lr',default=1e-4, type=float)
parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=False, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--nss_emlnet',default=False, type=bool)
parser.add_argument('--nss_norm',default=False, type=bool)
parser.add_argument('--l1',default=False, type=bool)
parser.add_argument('--lr_sched',default=False, type=bool)
parser.add_argument('--optim',default="Adam", type=str)

parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--step_size',default=5, type=int)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=1.0, type=float)
parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)

parser.add_argument('--batch_size',default=8, type=int)
parser.add_argument('--log_interval',default=5, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--model_val_path',default="enet_transformer.pt", type=str)
parser.add_argument('--clip_size',default=32, type=int)
parser.add_argument('--hidden_channels',default=256, type=int)
parser.add_argument('--get_multilevel_features',default=False, type=bool)
parser.add_argument('--train_path_data',default="/ssd_scratch/cvit/samyak/DHF1K/annotation", type=str)
parser.add_argument('--val_path_data',default="/ssd_scratch/cvit/samyak/DHF1K/val", type=str)

args = parser.parse_args()
print(args)

model = CombinedModel(
	hidden_channel=args.hidden_channel, 
    get_multilevel_features=args.get_multilevel_features
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# for (name, param) in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.size())

train_dataset = DHF1KDataset(args.train_path_data, args.clip_size, mode="train")
val_dataset = DHF1KDataset(args.val_path_data, args.clip_size, mode="val")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)


# parameter setting for fine-tuning
if args.optim == "Adam":
    params = list(filter(lambda p: p.requires_grad, model.parameters())) 
    optimizer = torch.optim.Adam(params, lr=args.lr)
elif args.optim=="Adagrad":
    optimizer = torch.optim.Adagrad(params, lr=args.lr)
elif args.optim=="SGD":
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
if args.lr_sched:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

print(device)

def train(model, optimizer, loader, epoch, device, args):
    model.train()
    tic = time.time()
    
    total_loss = AverageMeter()
    cur_loss = AverageMeter()

    for idx, (img_clips, gt_sal_clips) in enumerate(loader):
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((1,0,2,3,4))      
        gt_sal_clips = gt_sal_clips.permute((1,0,2,3))
         
        gt_sal_clips = gt_sal_clips.to(device)
        
        optimizer.zero_grad()
        pred_sal = model(img_clips)
        assert pred_sal.size() == gt_sal_clips.size()

        loss = loss_func(pred_sal, gt_sal_clips, args)
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
        cur_loss.update(loss.item())

        if idx%args.log_interval==(args.log_interval-1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss.avg, (time.time()-tic)/60))
            cur_loss.reset()
            sys.stdout.flush()

    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss.avg))
    sys.stdout.flush()

    return total_loss.avg

def validate(model, loader, epoch, device, args):
    model.eval()
    tic = time.time()
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    tic = time.time()
    for idx, (img_clips, gt_sal_clips) in enumerate(loader):
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((1,0,2,3,4))
        gt_sal_clips = gt_sal_clips.permute((1,0,2,3))
        
        pred_sal_clips = model(img_clips)
        
        for i in range(pred_sal_clips.size(0)):
            gt_sal = gt_sal_clips[i].squeeze(0).numpy()

            pred_sal = pred_sal_clips[i].cpu().squeeze(0).numpy()
            pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
            pred_sal = blur(pred_sal).unsqueeze(0).cuda()

            gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

            assert pred_sal.size() == gt_sal.size()

            loss = loss_func(pred_sal, gt_sal, args)
            cc_loss = cc(pred_sal, gt_sal)

            total_loss.update(loss.item())
            total_cc_loss.update(cc_loss.item())

    print('[{:2d}, val] avg_loss : {:.5f} cc_loss : {:.5f}, time : {:3f}'.format(epoch, total_loss.avg, total_cc_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()

    return total_loss.avg

best_model = None
for epoch in range(0, args.no_epochs):
    loss = train(model, optimizer, train_loader, epoch, device, args)
    
    with torch.no_grad():
        val_loss = validate(model, val_loader, epoch, device, args)
        if epoch == 0 :
            best_loss = val_loss
        if val_loss <= best_loss:
            best_loss = val_loss
            best_model = model
            print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
            if torch.cuda.device_count() > 1:    
                torch.save(model.module.state_dict(), args.model_val_path)
            else:
                torch.save(model.state_dict(), args.model_val_path)
    print()

    if args.lr_sched:
        scheduler.step()
