import sys
import os
import numpy as np
import cv2
import torch
# from model import TASED_v2
from model_hier import TASED_v2_hier as TASED_v2
from scipy.ndimage.filters import gaussian_filter
from loss import kldiv, cc, nss
import argparse
from utils import *
from os.path import join

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    ''' read frames in path_indata and generate frame-wise saliency maps in path_output '''
    path_indata = '/ssd_scratch/cvit/samyak/DHF1K/val'
    file_weight = './saved_models/transformer.pt'

    len_temporal = 32

    model = TASED_v2(use_transformer=True)

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

    if args.start_idx!=-1:
        _len = 0.25*len(list_indata)
        list_indata = list_indata[int((args.start_idx-1)*_len): int(args.start_idx*_len)]

    video_kldiv_loss = []
    video_cc_loss = []
    video_nss_loss = []
    # video_cnt = 0
    os.system('mkdir -p '+args.save_path)
    
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
                img = cv2.imread(os.path.join(path_indata, dname, 'images', list_frames[i]))
                img = cv2.resize(img, (384, 224))
                img = img[...,::-1]
                snippet.append(img)

                if i >= len_temporal-1:
                    clip = transform(snippet)

                    kldiv_loss, cc_loss, nss_loss = process(model, clip, path_indata, dname, list_frames[i], args)
                    total_kldiv_loss += kldiv_loss
                    total_nss_loss += nss_loss
                    total_cc_loss += cc_loss
                    total_cnt += 1

                    # print(clip.shape)
                    # process first (len_temporal-1) frames
                    if i < 2*len_temporal-2:
                        kldiv_loss, cc_loss, nss_loss = process(model, torch.flip(clip, [2]), path_indata, dname, list_frames[i-len_temporal+1], args)
                        total_kldiv_loss += kldiv_loss
                        total_nss_loss += nss_loss
                        total_cc_loss += cc_loss
                        total_cnt += 1

                    del snippet[0]

            video_kldiv_loss.append(total_kldiv_loss/total_cnt)
            video_cc_loss.append(total_cc_loss/total_cnt)
            video_nss_loss.append(total_nss_loss/total_cnt)

                # if total_cnt:
                #     print("idx {} CC: {}, kldiv: {}, nss: {}".format(total_cnt, total_cc_loss/total_cnt, total_kldiv_loss/total_cnt, total_nss_loss/total_cnt), end='\r')
        else:
            print (' more frames are needed')
        print("kldiv ", sum(video_kldiv_loss)/len(video_kldiv_loss))
        print("cc ", sum(video_cc_loss)/len(video_cc_loss))
        print("nss ", sum(video_nss_loss)/len(video_nss_loss))

    print("kldiv ", sum(video_kldiv_loss)/len(video_kldiv_loss))
    print("cc ", sum(video_cc_loss)/len(video_cc_loss))
    print("nss ", sum(video_nss_loss)/len(video_nss_loss))
    print()

    if args.start_idx!=-1:
        print("Non Average")
        print("kldiv ", sum(video_kldiv_loss))
        print("cc ", sum(video_cc_loss))
        print("nss ", sum(video_nss_loss))

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)

def process(model, clip, path_inpdata, dname, frame_no, args):
    ''' process one clip and save the predicted saliency map '''
    with torch.no_grad():
        smap = model(clip.to(device)).cpu().data[0]

    # smap = (smap.numpy()*255.).astype(np.int)/255.
    # smap = gaussian_filter(smap, sigma=7)
    # smap = (smap/np.max(smap)*255.).astype(np.uint8)


    
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

    if args.save:
        os.makedirs(join(args.save_path, dname), exist_ok=True)
        img_save(smap, join(args.save_path, dname, frame_no), normalize=True)


    fix = torch.FloatTensor(fix).unsqueeze(0).cuda()
    gt = torch.FloatTensor(gt).unsqueeze(0).cuda()
    smap = torch.FloatTensor(smap).unsqueeze(0).cuda()

    # print(smap.size(), gt.size(), fix.size())
    kldiv_loss = kldiv(smap, gt)
    cc_loss = cc(smap, gt)
    nss_loss = nss(smap, gt)

    return kldiv_loss, cc_loss, nss_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save',default=False, type=bool)
    parser.add_argument('--save_path',default='/ssd_scratch/cvit/samyak/Results/transformer', type=str)
    parser.add_argument('--start_idx',default=-1, type=int)
    args = parser.parse_args()
    print(args)
    main(args)

