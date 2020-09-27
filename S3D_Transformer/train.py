''' Kl - CC '''
''' Output 2 / 32 frames with S3D '''
''' Increasing time-step difference '''
''' ConvT -> Conv + Up '''
''' Finetuning using lr sched '''
''' Backbone with GRU/LSTM at highest level of heirarchy '''
''' Using Decoder to generate 32 gaussians '''

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
from dataloader import * 
from loss import *
import cv2
from model_hier import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs',default=50, type=int)
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
parser.add_argument('--nhead',default=4, type=int)
parser.add_argument('--num_encoder_layers',default=3, type=int)
parser.add_argument('--num_decoder_layers',default=3, type=int)
parser.add_argument('--transformer_in_channel',default=32, type=int)
parser.add_argument('--train_path_data',default="/ssd_scratch/cvit/samyak/DHF1K/annotation", type=str)
parser.add_argument('--val_path_data',default="/ssd_scratch/cvit/samyak/DHF1K/val", type=str)
parser.add_argument('--multi_frame',default=0, type=int)
parser.add_argument('--decoder_upsample',default=1, type=int)
parser.add_argument('--frame_no',default="last", type=str)
parser.add_argument('--load_weight',default="None", type=str)
parser.add_argument('--num_hier',default=3, type=int)
parser.add_argument('--dataset',default="DHF1KDataset", type=str)
parser.add_argument('--alternate',default=1, type=int)
parser.add_argument('--spatial_dim',default=-1, type=int)

args = parser.parse_args()
print(args)

file_weight = './S3D_kinetics400.pt'

if args.multi_frame==32:
    model = VideoSaliencyMultiModel(
    	transformer_in_channel=args.transformer_in_channel, 
    	use_transformer=True, 
        num_encoder_layers=args.num_encoder_layers, 
    	num_decoder_layers=args.num_decoder_layers, 
    	nhead=args.nhead,
        multiFrame=args.multi_frame,
        spatial_dim=args.spatial_dim
    )
else:
    ''' No transformer - S3D + ConvT + Up'''
    model = VideoSaliencyModel(
        transformer_in_channel=args.transformer_in_channel, 
        nhead=args.nhead,
        use_upsample=bool(args.decoder_upsample),
        num_hier=args.num_hier,
        num_clips=args.clip_size
    )

np.random.seed(0)
torch.manual_seed(0)

# for (name, param) in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.size())

if args.dataset == "DHF1KDataset":
    train_dataset = DHF1KDataset(args.train_path_data, args.clip_size, mode="train", multi_frame=args.multi_frame, alternate=args.alternate)
    val_dataset = DHF1KDataset(args.val_path_data, args.clip_size, mode="val", multi_frame=args.multi_frame, alternate=args.alternate)
else:
    train_dataset = Hollywood_UCFDataset(args.train_path_data, args.clip_size, mode="train", multi_frame=args.multi_frame)
    # print(len(train_dataset))
    val_dataset = Hollywood_UCFDataset(args.val_path_data, args.clip_size, mode="val", multi_frame=args.multi_frame)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

if os.path.isfile(file_weight):
    print ('loading weight file')
    weight_dict = torch.load(file_weight)
    model_dict = model.backbone.state_dict()
    for name, param in weight_dict.items():
        if 'module' in name:
            name = '.'.join(name.split('.')[1:])
        if 'base.' in name:
            bn = int(name.split('.')[1])
            sn_list = [0, 5, 8, 14]
            sn = sn_list[0]
            if bn >= sn_list[1] and bn < sn_list[2]:
                sn = sn_list[1]
            elif bn >= sn_list[2] and bn < sn_list[3]:
                sn = sn_list[2]
            elif bn >= sn_list[3]:
                sn = sn_list[3]
            name = '.'.join(name.split('.')[2:])
            name = 'base%d.%d.'%(sn_list.index(sn)+1, bn-sn)+name
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print (' size? ' + name, param.size(), model_dict[name].size())
        else:
            print (' name? ' + name)

    print (' loaded')
    model.backbone.load_state_dict(model_dict)
else:
    print ('weight file?')

if args.load_weight!="None":
    print("Loading weights: ",args.load_weight)
    model.load_state_dict(torch.load(args.load_weight))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# parameter setting for fine-tuning
params = list(filter(lambda p: p.requires_grad, model.parameters())) 
optimizer = torch.optim.Adam(params, lr=args.lr)

# if args.optim=="Adagrad":
#     optimizer = torch.optim.Adagrad(params, lr=args.lr)
# if args.optim=="SGD":
#     optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
# if args.lr_sched:
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

print(device)

def train(model, optimizer, loader, epoch, device, args):
    model.train()
    tic = time.time()
    
    total_loss = AverageMeter()
    cur_loss = AverageMeter()

    for idx, (img_clips, gt_sal) in enumerate(loader):
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0,2,1,3,4))
        gt_sal = gt_sal.to(device)
        
        optimizer.zero_grad()
        if args.multi_frame:
            idx_frame = np.random.randint(0, args.multi_frame)
            # print(idx_frame)
            gt_sal = gt_sal[:,idx_frame,...]
            pred_sal = model(img_clips, idx_frame)
            pred_sal = pred_sal.squeeze(1)
            # print(pred_sal.shape, gt_sal.shape)
        else:
            pred_sal = model(img_clips)
        assert pred_sal.size() == gt_sal.size()

        loss = loss_func(pred_sal, gt_sal, args)
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
    total_sim_loss = AverageMeter()
    tic = time.time()
    for idx, (img_clips, gt_sal) in enumerate(loader):
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0,2,1,3,4))
        
        pred_sal = model(img_clips)
        
        gt_sal = gt_sal.squeeze(0).numpy()

        pred_sal = pred_sal.cpu().squeeze(0).numpy()
        pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
        pred_sal = blur(pred_sal).unsqueeze(0).cuda()

        gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

        assert pred_sal.size() == gt_sal.size()

        loss = loss_func(pred_sal, gt_sal, args)
        cc_loss = cc(pred_sal, gt_sal)
        sim_loss = similarity(pred_sal, gt_sal)

        total_loss.update(loss.item())
        total_cc_loss.update(cc_loss.item())
        total_sim_loss.update(sim_loss.item())

    print('[{:2d}, val] avg_loss : {:.5f} cc_loss : {:.5f} sim_loss : {:.5f}, time : {:3f}'.format(epoch, total_loss.avg, total_cc_loss.avg, total_sim_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()

    return total_loss.avg

def validateMulti(model, loader, epoch, device, args):
    model.eval()
    tic = time.time()
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    tic = time.time()
    for idx, (img_clips, gt_sal_clips) in enumerate(loader):
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0,2,1,3,4))
        pred_sal_clips = model(img_clips, -1)
        # print(pred_sal_clips.shape, gt_sal_clips.shape)
        
        for i in range(pred_sal_clips.size(1)):
            gt_sal = gt_sal_clips[:,i,:,:].squeeze(0).numpy()

            pred_sal = pred_sal_clips[:,i,:,:].cpu().squeeze(0).numpy()
            pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
            pred_sal = blur(pred_sal).unsqueeze(0).cuda()

            gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

            assert pred_sal.size() == gt_sal.size()

            loss = loss_func(pred_sal, gt_sal, args)
            cc_loss = cc(pred_sal, gt_sal)
            sim_loss = similarity(pred_sal, gt_sal)

            total_loss.update(loss.item())
            total_cc_loss.update(cc_loss.item())
            total_sim_loss.update(sim_loss.item())

    print('[{:2d}, val] avg_loss : {:.5f} cc_loss : {:.5f} sim_loss : {:.5f}, time : {:3f}'.format(epoch, total_loss.avg, total_cc_loss.avg, total_sim_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()

    return total_loss.avg

best_model = None
for epoch in range(0, args.no_epochs):
    loss = train(model, optimizer, train_loader, epoch, device, args)
    
    with torch.no_grad():
        if args.multi_frame:
            val_loss = validateMulti(model, val_loader, epoch, device, args)
        else:
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
