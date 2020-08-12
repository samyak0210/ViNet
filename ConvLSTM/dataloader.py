import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class DHF1KDataset(Dataset):
    def __init__(self, path_data, len_snippet, mode="train"):
        ''' mode: train, val '''
        self.path_data = path_data
        self.len_snippet = len_snippet
        self.mode = mode
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        for v in os.listdir(path_data):
            for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))-self.len_snippet, self.len_snippet):
                self.list_num_frame.append((v, i))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        (file_name, start_idx) = self.list_num_frame[idx]

        path_clip = os.path.join(self.path_data, file_name, 'images')
        path_annt = os.path.join(self.path_data, file_name, 'maps')

        clip_img = []
        clip_gt = []
        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+i+1))).convert('RGB')
            gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+i+1))).convert('L'))
            gt = gt.astype('float')
            
            if self.mode == "train":
                gt = cv2.resize(gt, (256, 256))
            
            if np.max(gt) > 1.0:
                gt = gt / 255.0

            clip_img.append(self.img_transform(img))            
            clip_gt.append(torch.FloatTensor(gt))
            
        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))

        return clip_img, clip_gt
