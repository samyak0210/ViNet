import sys
import os
import numpy as np
import cv2
import torch
from model_hier import VideoSaliencyMultiModel, VideoSaliencyMultiModelParallel
from scipy.ndimage.filters import gaussian_filter
from loss import kldiv, cc, nss
import argparse
from dataloader import *
from torch.utils.data import DataLoader
from dataloader import DHF1KDataset
from utils import *
import time
from tqdm import tqdm
from torchvision import transforms, utils
from os.path import join

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validateMulti(model, loader, epoch, device, args):
	model.eval()
	tic = time.time()
	total_loss = AverageMeter()
	total_cc_loss = AverageMeter()
	total_sim_loss = AverageMeter()
	tic = time.time()
	for (img_clips, start_idxs, d_names, sizes) in tqdm(loader):
		img_clips = img_clips.to(device)
		img_clips = img_clips.permute((0,2,1,3,4))
		
		pred_sal_clips = model(img_clips, -1)
		# continue
		for i in range(pred_sal_clips.size(0)):
			pred_sal_clip = pred_sal_clips[i]
			start_idx = start_idxs[i]
			dname = d_names[i]

			os.makedirs(join(args.save_path, dname), exist_ok=True)
			for j in range(pred_sal_clip.size(0)):
				pred_sal = pred_sal_clip[j].cpu().numpy()	

				if args.alternate!=1 and j>0:
					pred_sal_middle = (pred_sal_clip[j-1].cpu().numpy() + pred_sal_clip[j].cpu().numpy())/2
					pred_sal_middle = cv2.resize(pred_sal_middle, (sizes[0], sizes[1]))
					pred_sal_middle = blur(pred_sal_middle)
					img_save(pred_sal_middle, join(args.save_path, dname, '%04d.png'%(start_idx+args.alternate * j)), normalize=True)

				pred_sal = cv2.resize(pred_sal, (sizes[0], sizes[1]))
				pred_sal = blur(pred_sal)

				

				img_save(pred_sal, join(args.save_path, dname, '%04d.png'%(start_idx+args.alternate * j+1)), normalize=True)
				if j==pred_sal_clip.size(0)-1:
					img_save(pred_sal, join(args.save_path, dname, '%04d.png'%(start_idx+args.alternate * j+2)), normalize=True)
			
def validate(args):
	''' read frames in path_indata and generate frame-wise saliency maps in path_output '''
	# optional two command-line arguments
	file_weight = args.file_weight

	len_temporal = 32

	model = VideoSaliencyMultiModelParallel(
		transformer_in_channel=args.transformer_in_channel, 
		use_transformer=True,
		num_encoder_layers=args.num_encoder_layers, 
		num_decoder_layers=args.num_decoder_layers, 
		nhead=args.nhead,
		multiFrame=args.multi_frame,
	)

	model.load_state_dict(torch.load(file_weight))
	# if torch.cuda.device_count() > 1:
	#     print("Let's use", torch.cuda.device_count(), "GPUs!")
	#     model.decoder = nn.DataParallel(model.decoder)

	model = model.to(device)
	torch.backends.cudnn.benchmark = False
	model.eval()

	# iterate over the path_indata directory
	val_dataset = DHF1KMultiSave(args.val_path_data, args.clip_size, start_idx=args.start_idx, num_parts=args.num_parts, alternate=args.alternate)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4*4)
	with torch.no_grad():
		validateMulti(model, val_loader, 0, device, args)


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
	sz = img.size
	img = img_transform(img)
	return img, sz

def blur(img):
	k_size = 11
	bl = cv2.GaussianBlur(img,(k_size,k_size),0)
	return torch.FloatTensor(bl)

def process(model, clip, path_inpdata, dname, frame_no, args, img_size):
	''' process one clip and save the predicted saliency map '''
	with torch.no_grad():
		smap = model(clip.to(device)).cpu()
	# print(smap.size())
	for i in range(len(frame_no)):
		s_map = smap[:,i,:,:].squeeze(0).numpy()
		s_map = cv2.resize(s_map, (img_size[1], img_size[0]))
		s_map = blur(s_map)

		img_save(s_map, join(args.save_path, dname, frame_no[i]), normalize=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_weight',default="./saved_models/alternate_multi_frame.pt", type=str)
	parser.add_argument('--nhead',default=4, type=int)
	parser.add_argument('--num_encoder_layers',default=3, type=int)
	parser.add_argument('--transformer_in_channel',default=32, type=int)
	parser.add_argument('--save_path',default='/ssd_scratch/cvit/samyak/Results/alternate', type=str)
	parser.add_argument('--start_idx',default=-1, type=int)
	parser.add_argument('--num_parts',default=4, type=int)
	parser.add_argument('--multi_frame',default=32, type=int)
	parser.add_argument('--decoder_upsample',default=1, type=int)
	parser.add_argument('--num_decoder_layers',default=3, type=int)
	parser.add_argument('--val_path_data',default="/ssd_scratch/cvit/samyak/DHF1K/val", type=str)
	parser.add_argument('--clip_size',default=32, type=int)
	parser.add_argument('--batch_size',default=1, type=int)
	parser.add_argument('--alternate',default=2, type=int)

	
	args = parser.parse_args()
	print(args)
	validate(args)

