import sys
import os
import numpy as np
import cv2
import torch
from model_hier import TASED_v2_hier as TASED_v2
from scipy.ndimage.filters import gaussian_filter
from loss import kldiv, cc, nss
import argparse

from torch.utils.data import DataLoader
from dataloader import DHF1KDataset
from utils import *
import time
from tqdm import tqdm
from torchvision import transforms, utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(args):
    ''' read frames in path_indata and generate frame-wise saliency maps in path_output '''
    # optional two command-line arguments
    path_indata = '/ssd_scratch/cvit/navyasri/DHF1K/val'
    file_weight = args.file_weight

    len_temporal = 32

    model = TASED_v2(
        transformer_in_channel=args.transformer_in_channel, 
        use_transformer=True, 
        num_encoder_layers=args.num_encoder_layers, 
        nhead=args.nhead
    )
    # model.load_state_dict(torch.load(file_weight))
    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight, map_location=device)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    # iterate over the path_indata directory
    list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()
    video_kldiv_loss = []
    video_cc_loss = []
    video_nss_loss = []
    for dname in list_indata:
        print ('processing ' + dname, flush=True)
        list_frames = [f for f in os.listdir(os.path.join(path_indata, dname, 'images')) if os.path.isfile(os.path.join(path_indata, dname, 'images', f))]
        list_frames.sort()

        # process in a sliding window fashion
        if len(list_frames) >= 2*len_temporal-1:

            total_kldiv_loss = 0.0
            total_cc_loss = 0.0
            total_nss_loss = 0.0
            total_cnt = 0
            snippet = []
            for i in range(len(list_frames)):
                torch_img = torch_transform(os.path.join(path_indata, dname, 'images', list_frames[i]))
                snippet.append(torch_img)

                if i >= len_temporal-1:
                    clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                    clip = clip.permute((0,2,1,3,4))

                    kldiv_loss, cc_loss, nss_loss = process(model, clip, path_indata, dname, list_frames[i])
                    total_kldiv_loss += kldiv_loss
                    total_nss_loss += nss_loss
                    total_cc_loss += cc_loss
                    total_cnt += 1

                    # process first (len_temporal-1) frames
                    if i < 2*len_temporal-2:
                        kldiv_loss, cc_loss, nss_loss = process(model, torch.flip(clip, [2]), path_indata, dname, list_frames[i-len_temporal+1])
                        total_kldiv_loss += kldiv_loss
                        total_nss_loss += nss_loss
                        total_cc_loss += cc_loss
                        total_cnt += 1

                    del snippet[0]

            video_kldiv_loss.append(total_kldiv_loss/total_cnt)
            video_cc_loss.append(total_cc_loss/total_cnt)
            video_nss_loss.append(total_nss_loss/total_cnt)

        else:
            print (' more frames are needed')
        print("kldiv ", sum(video_kldiv_loss)/len(video_kldiv_loss))
        print("cc ", sum(video_cc_loss)/len(video_cc_loss))
        print("nss ", sum(video_nss_loss)/len(video_nss_loss))

    print("kldiv ", sum(video_kldiv_loss)/len(video_kldiv_loss))
    print("cc ", sum(video_cc_loss)/len(video_cc_loss))
    print("nss ", sum(video_nss_loss)/len(video_nss_loss))

def torch_transform(path):
    img_transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
    ])
    img = Image.open(path).convert('RGB')
    img = img_transform(img)
    return img

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)

def process(model, clip, path_inpdata, dname, frame_no):
    ''' process one clip and save the predicted saliency map '''
    with torch.no_grad():
        smap = model(clip.to(device)).cpu().data[0]
    
    gt = cv2.imread(os.path.join(path_inpdata, dname, 'maps', frame_no), 0)
    gt = gt.astype('float')
    if np.max(gt) > 1.0:
        gt = gt / 255.0
    
    fix = cv2.imread(os.path.join(path_inpdata, dname, 'fixation', frame_no), 0)
    fix = fix.astype('float')
    fix = (fix > 0.5).astype('float')
    
    smap = smap.numpy()
    smap = cv2.resize(smap, (gt.shape[1], gt.shape[0]))
    smap = blur(smap)


    fix = torch.FloatTensor(fix).unsqueeze(0).cuda()
    gt = torch.FloatTensor(gt).unsqueeze(0).cuda()
    smap = torch.FloatTensor(smap).unsqueeze(0).cuda()

    assert smap.size() == gt.size() and smap.size() == fix.size()

    kldiv_loss = kldiv(smap, gt)
    cc_loss = cc(smap, gt)
    nss_loss = nss(smap, gt)

    return kldiv_loss, cc_loss, nss_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_weight',default="./saved_models/adam_transformer_fixed_sample_more_epochs.pt", type=str)
    parser.add_argument('--use_transformer',default=True, type=bool)
    parser.add_argument('--nhead',default=4, type=int)
    parser.add_argument('--num_encoder_layers',default=3, type=int)
    parser.add_argument('--transformer_in_channel',default=32, type=int)

    args = parser.parse_args()
    print(args)
    validate(args)

