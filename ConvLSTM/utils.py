import torch
import torch.nn as nn
from loss import *
import cv2

def loss_func(pred_map, gt, args):
    ''' pred_map: ClxBxHxW '''
    loss = torch.FloatTensor([0.0]).cuda()
    criterion = nn.L1Loss()
    clip_size = pred_map.size(0)
    for i in range(clip_size):
        if args.kldiv:
            loss += args.kldiv_coeff * kldiv(pred_map[i], gt[i])
        if args.cc:
            loss += args.cc_coeff * cc(pred_map[i], gt[i])
        if args.l1:
            loss += args.l1_coeff * criterion(pred_map[i], gt[i])
        if args.sim:
            loss += args.sim_coeff * similarity(pred_map[i], gt[i])
    loss = loss / clip_size
    return loss

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)