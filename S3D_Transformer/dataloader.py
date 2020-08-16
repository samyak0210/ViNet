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
        ''' mode: train, val, perframe '''
        self.path_data = path_data
        self.len_snippet = len_snippet
        self.mode = mode
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        if self.mode == "train":
            self.video_names = os.listdir(path_data)
            self.list_num_frame = [len(os.listdir(os.path.join(path_data,d,'images'))) for d in self.video_names]
        elif self.mode=="val":
            self.list_num_frame = []
            for v in os.listdir(path_data):
                for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))-self.len_snippet, self.len_snippet):
                    self.list_num_frame.append((v, i))
        else:
            self.list_num_frame = []
            for v in os.listdir(path_data):
                for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))-self.len_snippet):
                    self.list_num_frame.append((v, i, False))
                for i in range(0, len_snippet):
                    self.list_num_frame.append((v, i+len_snippet-1, True))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        isFlip = False
        # print(self.mode)
        if self.mode == "train":
            file_name = self.video_names[idx]
            start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)
        elif self.mode == "val":
            (file_name, start_idx) = self.list_num_frame[idx]
        else:
            (file_name, start_idx, isFlip) = self.list_num_frame[idx]

        path_clip = os.path.join(self.path_data, file_name, 'images')
        path_annt = os.path.join(self.path_data, file_name, 'maps')

        clip_img = []
        for i in range(self.len_snippet):
            if not isFlip:
                img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+i+1))).convert('RGB')
            else:
                img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx-i+1))).convert('RGB')
            clip_img.append(self.img_transform(img))
            
        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))

        if not isFlip:
            gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+self.len_snippet))).convert('L'))
        else:
            gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx-self.len_snippet+2))).convert('L'))
        gt = gt.astype('float')
        
        if self.mode == "train":
            gt = cv2.resize(gt, (384, 224))
        
        if np.max(gt) > 1.0:
            gt = gt / 255.0

        return clip_img, torch.FloatTensor(gt)

class DHF1KDatasetMultiFrame(Dataset):
    def __init__(self, path_data, len_snippet, out_clips=32, mode="train"):
        ''' mode: train, val, perframe '''
        self.path_data = path_data
        self.out_clips = out_clips
        self.len_snippet = len_snippet
        self.mode = mode
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        if self.mode == "train":
            self.video_names = os.listdir(path_data)
            self.list_num_frame = [len(os.listdir(os.path.join(path_data,d,'images'))) for d in self.video_names]
        elif self.mode=="val":
            self.list_num_frame = []
            for v in os.listdir(path_data):
                for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))-self.len_snippet, self.len_snippet):
                    self.list_num_frame.append((v, i))
        else:
            self.list_num_frame = []
            for v in os.listdir(path_data):
                num_frame = len(os.listdir(os.path.join(path_data,v,'images')))
                for i in range(0, num_frame-self.len_snippet, self.len_snippet):
                    self.list_num_frame.append((v, i, 32))
                
                if num_frame%self.len_snippet:
                    self.list_num_frame.append((v, num_frame - self.len_snippet - 1, num_frame%self.len_snippet))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        # print(self.mode)
        if self.mode == "train":
            file_name = self.video_names[idx]
            start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)
        elif self.mode == "val":
            (file_name, start_idx) = self.list_num_frame[idx]
        else:
            (file_name, start_idx, frames_included) = self.list_num_frame[idx]

        path_clip = os.path.join(self.path_data, file_name, 'images')
        path_annt = os.path.join(self.path_data, file_name, 'maps')

        clip_img = []
        clip_gt = []
        
        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+i+1))).convert('RGB')
            gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+i+1))).convert('L'))
            gt = gt.astype('float')
            
            if self.mode == "train":
                gt = cv2.resize(gt, (384, 224))
            
            if np.max(gt) > 1.0:
                gt = gt / 255.0

            clip_img.append(self.img_transform(img))
            clip_gt.append(torch.FloatTensor(gt))
            
        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))

        return clip_img, clip_gt

class DHF1KDatasetDualFrame(Dataset):
    def __init__(self, path_data, len_snippet, out_clips=32, mode="train"):
        ''' mode: train, val, perframe '''
        self.path_data = path_data
        self.out_clips = out_clips
        self.len_snippet = len_snippet
        self.mode = mode
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        if self.mode == "train":
            self.video_names = os.listdir(path_data)
            self.list_num_frame = [len(os.listdir(os.path.join(path_data,d,'images'))) for d in self.video_names]
        elif self.mode=="val":
            self.list_num_frame = []
            for v in os.listdir(path_data):
                for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))-self.len_snippet, self.len_snippet):
                    self.list_num_frame.append((v, i))
        else:
            self.list_num_frame = []
            for v in os.listdir(path_data):
                num_frame = len(os.listdir(os.path.join(path_data,v,'images')))
                for i in range(0, num_frame-self.len_snippet, self.len_snippet):
                    self.list_num_frame.append((v, i, 32))
                
                if num_frame%self.len_snippet:
                    self.list_num_frame.append((v, num_frame - self.len_snippet - 1, num_frame%self.len_snippet))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        # print(self.mode)
        if self.mode == "train":
            file_name = self.video_names[idx]
            start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)
        elif self.mode == "val":
            (file_name, start_idx) = self.list_num_frame[idx]
        else:
            (file_name, start_idx, frames_included) = self.list_num_frame[idx]

        path_clip = os.path.join(self.path_data, file_name, 'images')
        path_annt = os.path.join(self.path_data, file_name, 'maps')

        clip_img = []
        clip_gt = []
        
        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+i+1))).convert('RGB')
            
            if i==0 or i==self.len_snippet-1:
                gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+i+1))).convert('L'))
                gt = gt.astype('float')
                
                if self.mode == "train":
                    gt = cv2.resize(gt, (384, 224))
                
                if np.max(gt) > 1.0:
                    gt = gt / 255.0
                clip_gt.append(torch.FloatTensor(gt))

            clip_img.append(self.img_transform(img))
            
        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))

        return clip_img, clip_gt
