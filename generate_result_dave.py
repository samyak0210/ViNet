import sys
import os
import numpy as np
import cv2
import torch
from model import *
from scipy.ndimage.filters import gaussian_filter
from loss import kldiv, cc, nss
import argparse

from torch.utils.data import DataLoader
from dataloader import DHF1KDataset
from utils import *
import time
from tqdm import tqdm
from torchvision import transforms, utils
from os.path import join
import torchaudio
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def read_sal_text_dave(json_file):
	test_list = {'names': [], 'nframes': [], 'fps': []}
	with open(json_file,'r') as f:
		_dic = json.load(f)
		for name in _dic:
			test_list['names'].append(name)
			test_list['nframes'].append(0)
			test_list['fps'].append(float(_dic[name]))
	return test_list	

def make_dataset(annotation_path, audio_path, gt_path, vox=False, json_file=None):
	if json_file is None:
		data = read_sal_text(annotation_path)
	else:
		data = read_sal_text_dave(json_file)
	video_names = data['names']
	video_nframes = data['nframes']
	video_fps = data['fps']
	dataset = []
	audiodata= {}
	for i in range(len(video_names)):
		if i % 100 == 0:
			print('dataset loading [{}/{}]'.format(i, len(video_names)))

		n_frames = len(os.listdir(join(gt_path, video_names[i], 'maps')))
		if n_frames <= 1:
			print("Less frames")
			continue

		begin_t = 1
		end_t = n_frames

		audio_wav_path = os.path.join(audio_path,video_names[i],video_names[i]+'.wav')
		if not os.path.exists(audio_wav_path):
			print("Not exists", audio_wav_path)
			continue
		if not vox:
			[audiowav,Fs] = torchaudio.load(audio_wav_path, normalization=False)
			audiowav = audiowav * (2 ** -23)
		else:
			Fs, audiowav = wavfile.read(audio_wav_path)
			audiowav = torch.from_numpy(audiowav).unsqueeze(0)
		n_samples = Fs/float(video_fps[i])
		starts=np.zeros(n_frames+1, dtype=int)
		ends=np.zeros(n_frames+1, dtype=int)
		starts[0]=0
		ends[0]=0
		for videoframe in range(1,n_frames+1):
			startemp=max(0,((videoframe-1)*(1.0/float(video_fps[i]))*Fs)-n_samples/2)
			starts[videoframe] = int(startemp)
			endtemp=min(audiowav.shape[1],abs(((videoframe-1)*(1.0/float(video_fps[i]))*Fs)+n_samples/2))
			ends[videoframe] = int(endtemp)

		audioinfo = {
			'audiopath': audio_path,
			'video_id': video_names[i],
			'Fs' : Fs,
			'wav' : audiowav,
			'starts': starts,
			'ends' : ends
		}

		audiodata[video_names[i]] = audioinfo

	return audiodata

def get_audio_feature(audioind, audiodata, args, start_idx):
	len_snippet = args.clip_size
	max_audio_Fs = 22050
	min_video_fps = 10
	max_audio_win = int(max_audio_Fs / min_video_fps * 32)

	audioexcer  = torch.zeros(1,max_audio_win)
	valid = {}
	valid['audio']=0

	if audioind in audiodata:

		excerptstart = audiodata[audioind]['starts'][start_idx+1]
		if start_idx+len_snippet >= len(audiodata[audioind]['ends']):
			print("Exceeds size", audioind)
			sys.stdout.flush()
			excerptend = audiodata[audioind]['ends'][-1]
		else:
			excerptend = audiodata[audioind]['ends'][start_idx+len_snippet]	
		try:
			valid['audio'] = audiodata[audioind]['wav'][:, excerptstart:excerptend+1].shape[1]
		except:
			pass
		audioexcer_tmp = audiodata[audioind]['wav'][:, excerptstart:excerptend+1]
		if (valid['audio']%2)==0:
			audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2))] = \
				torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
		else:
			audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2)+1)] = \
				torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
	audio_feature = audioexcer.view(1, 1,-1,1)
	return audio_feature

def validate(args):
	path_indata = args.path_indata
	file_weight = args.file_weight

	len_temporal = args.clip_size

	if args.use_sound:
		model = VideoAudioSaliencyModel(
			transformer_in_channel=args.transformer_in_channel, 
			nhead=args.nhead,
			use_upsample=bool(args.decoder_upsample),
			num_hier=args.num_hier,
			num_clips=args.clip_size   
		)	
	else:
		model = VideoSaliencyModel(
			transformer_in_channel=args.transformer_in_channel, 
			nhead=args.nhead,
			use_upsample=bool(args.decoder_upsample),
			num_hier=args.num_hier,
			num_clips=args.clip_size   
		)

	model.load_state_dict(torch.load(file_weight))

	model = model.to(device)
	torch.backends.cudnn.benchmark = False
	model.eval()

	list_indata = []
	if args.dataset=='DIEM':
		file_name = 'DIEM_list_test_fps.txt'
	else:
		file_name = '{}_list_test_{}_fps.txt'.format(args.dataset, args.split)
	

	with open(join(args.path_indata, 'DAVE_fold_lists', file_name), 'r') as f:
		for line in f.readlines():
			name = line.split(' ')[0].strip()
			list_indata.append(name)
	list_indata.sort()
	print(list_indata, len(list_indata))

	if args.use_sound:
		json_file = '{}_fps_map.json'.format(args.dataset)
		audiodata = make_dataset(
			join(args.path_indata, 'fold_lists', file_name), 
			join(args.path_indata, 'video_audio', args.dataset),
			join(args.path_indata, 'annotations', args.dataset),
			json_file=join(args.path_indata, 'DAVE_fold_lists', json_file)
		)

	if args.start_idx!=-1:
		_len = (1.0/float(args.num_parts))*len(list_indata)
		list_indata = list_indata[int((args.start_idx-1)*_len): int(args.start_idx*_len)]

	for dname in list_indata:
		print ('processing ' + dname, flush=True)
		list_frames = [f for f in os.listdir(os.path.join(path_indata, 'video_frames', args.dataset, dname)) if os.path.isfile(os.path.join(path_indata, 'video_frames', args.dataset, dname, f))]        
		list_frames.sort()
		os.makedirs(join(args.save_path, dname), exist_ok=True)

		if len(list_frames) >= 2*len_temporal-1:

			snippet = []
			for i in range(len(list_frames)):
				torch_img, img_size = torch_transform(os.path.join(path_indata, 'video_frames', args.dataset, dname, list_frames[i]))

				snippet.append(torch_img)
				
				if i >= len_temporal-1:
					clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
					clip = clip.permute((0,2,1,3,4))

					audio_feature = None
					if args.use_sound:
						audio_feature = get_audio_feature(dname, audiodata, args, i-len_temporal+1)
					process(model, clip, path_indata, dname, list_frames[i], args, img_size, audio_feature=audio_feature)

					if i < 2*len_temporal-2:
						if args.use_sound:
							audio_feature = torch.flip(audio_feature, [2])
						process(model, torch.flip(clip, [2]), path_indata, dname, list_frames[i-len_temporal+1], args, img_size, audio_feature=audio_feature)

					del snippet[0]
		else:
			print (' more frames are needed')

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

def process(model, clip, path_inpdata, dname, frame_no, args, img_size, audio_feature=None):
	with torch.no_grad():
		if audio_feature is None:
			smap = model(clip.to(device)).cpu().data[0]
		else:
			smap = model(clip.to(device), audio_feature.to(device)).cpu().data[0]
	
	smap = smap.numpy()
	smap = cv2.resize(smap, (img_size[0], img_size[1]))
	smap = blur(smap)

	img_save(smap, join(args.save_path, dname, frame_no), normalize=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_weight',default="./saved_models/AViNet_Dave.pt", type=str)
	parser.add_argument('--nhead',default=4, type=int)
	parser.add_argument('--num_encoder_layers',default=3, type=int)
	parser.add_argument('--transformer_in_channel',default=512, type=int)
	parser.add_argument('--save_path',default='/ssd_scratch/cvit/samyak/Results/diem_test', type=str)
	parser.add_argument('--start_idx',default=-1, type=int)
	parser.add_argument('--num_parts',default=4, type=int)
	parser.add_argument('--split',default=1, type=int)
	parser.add_argument('--path_indata',default='/ssd_scratch/cvit/samyak/data/', type=str)
	parser.add_argument('--dataset',default='DIEM', type=str)
	parser.add_argument('--multi_frame',default=0, type=int)
	parser.add_argument('--decoder_upsample',default=1, type=int)
	parser.add_argument('--num_decoder_layers',default=-1, type=int)
	parser.add_argument('--num_hier',default=3, type=int)
	parser.add_argument('--clip_size',default=32, type=int)
	parser.add_argument('--use_sound',default=False, type=bool)
	
	args = parser.parse_args()
	print(args)
	validate(args)

