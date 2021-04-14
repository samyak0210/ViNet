import os
from os.path import join
import csv
import cv2, copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchaudio
import sys
from scipy.io import wavfile
import json

def read_sal_text(txt_file):
	test_list = {'names': [], 'nframes': [], 'fps': []}
	with open(txt_file,'r') as f:
		for line in f:
			word=line.strip().split()
			test_list['names'].append(word[0])
			test_list['nframes'].append(word[1])
			test_list['fps'].append(word[2])
	return test_list

def read_sal_text_dave(json_file):
	test_list = {'names': [], 'nframes': [], 'fps': []}
	with open(json_file,'r') as f:
		_dic = json.load(f)
		for name in _dic:
			# word=line.strip().split()
			test_list['names'].append(name)
			test_list['nframes'].append(0)
			test_list['fps'].append(float(_dic[name]))
	return test_list	

def make_dataset(annotation_path, audio_path, gt_path, json_file=None):
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
		[audiowav,Fs] = torchaudio.load(audio_wav_path, normalization=False)
		audiowav = audiowav * (2 ** -23)
		
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

def get_audio_feature(audioind, audiodata, clip_size, start_idx):
	len_snippet = clip_size
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
	else:
		print(audioind, "not present in data")
	audio_feature = audioexcer.view(1,-1,1)
	return audio_feature

class SoundDatasetLoader(Dataset):
	def __init__(self, len_snippet, dataset_name='DIEM', split=1, mode='train', use_sound=False, use_vox=False):
		''' mode: train, val, save '''
		path_data = '/ssd_scratch/cvit/samyak/data/'
		self.path_data = path_data
		self.use_vox = use_vox
		self.use_sound = use_sound
		self.mode = mode
		self.len_snippet = len_snippet
		self.img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		self.list_num_frame = []
		self.dataset_name = dataset_name
		if dataset_name=='DIEM':
			file_name = 'DIEM_list_{}_fps.txt'.format(mode)
		else:
			file_name = '{}_list_{}_{}_fps.txt'.format(dataset_name, mode, split)
		
		self.list_indata = []
		with open(join(self.path_data, 'fold_lists', file_name), 'r') as f:
		# with open(join(self.path_data, 'fold_lists', file_name), 'r') as f:
			for line in f.readlines():
				name = line.split(' ')[0].strip()
				self.list_indata.append(name)

		self.list_indata.sort()	
		print(self.mode, len(self.list_indata))
		if self.mode=='train':
			self.list_num_frame = [len(os.listdir(os.path.join(path_data,'annotations', dataset_name, v, 'maps'))) for v in self.list_indata]
		
		elif self.mode == 'test' or self.mode == 'val': 
			print("val set")
			for v in self.list_indata:
				frames = os.listdir(join(path_data, 'annotations', dataset_name, v, 'maps'))
				frames.sort()
				for i in range(0, len(frames)-self.len_snippet,  2*self.len_snippet):
					if self.check_frame(join(path_data, 'annotations', dataset_name, v, 'maps', 'eyeMap_%05d.jpg'%(i+self.len_snippet))):
						self.list_num_frame.append((v, i))

		max_audio_Fs = 22050
		min_video_fps = 10
		self.max_audio_win = int(max_audio_Fs / min_video_fps * 32)
		# assert use_sound ^ use_vox == True, (use_sound, use_vox)
		if use_sound or use_vox:
			if self.mode=='val':
				file_name = file_name.replace('val', 'test')
			json_file = '{}_fps_map.json'.format(self.dataset_name)
			self.audiodata = make_dataset(
					join(self.path_data, 'fold_lists', file_name), 
					join(self.path_data, 'video_audio', self.dataset_name),
					join(self.path_data, 'annotations', self.dataset_name),
					# vox=use_vox,
					# json_file=join(self.path_data, 'DAVE_fold_lists', json_file)
				)

	def check_frame(self, path):
		img = cv2.imread(path, 0)
		return img.max()!=0

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		# print(self.mode)
		if self.mode == "train":
			video_name = self.list_indata[idx]
			while 1:
				start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)
				if self.check_frame(join(self.path_data, 'annotations', self.dataset_name, video_name, 'maps', 'eyeMap_%05d.jpg'%(start_idx+self.len_snippet))):
					break
				else:
					print("No saliency defined in train dataset")
					sys.stdout.flush()

		elif self.mode == "test" or self.mode == "val":
			(video_name, start_idx) = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, 'video_frames', self.dataset_name, video_name)
		path_annt = os.path.join(self.path_data, 'annotations', self.dataset_name, video_name, 'maps')

		if self.use_sound:
			audio_feature = get_audio_feature(video_name, self.audiodata, self.len_snippet, start_idx)

		clip_img = []
		
		for i in range(self.len_snippet):
			img = Image.open(join(path_clip, 'img_%05d.jpg'%(start_idx+i+1))).convert('RGB')
			sz = img.size		
			clip_img.append(self.img_transform(img))
			
		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
		
		gt = np.array(Image.open(join(path_annt, 'eyeMap_%05d.jpg'%(start_idx+self.len_snippet))).convert('L'))
		gt = gt.astype('float')
		
		if self.mode == "train":
			gt = cv2.resize(gt, (384, 224))

		if np.max(gt) > 1.0:
			gt = gt / 255.0
		assert gt.max()!=0, (start_idx, video_name)
		if self.use_sound or self.use_vox:
			return clip_img, gt, audio_feature
		return clip_img, gt

class DHF1KDataset(Dataset):
	def __init__(self, path_data, len_snippet, mode="train", multi_frame=0, alternate=1):
		''' mode: train, val, save '''
		self.path_data = path_data
		self.len_snippet = len_snippet
		self.mode = mode
		self.multi_frame = multi_frame
		self.alternate = alternate
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
				for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))- self.alternate * self.len_snippet, 4*self.len_snippet):
					self.list_num_frame.append((v, i))
		else:
			self.list_num_frame = []
			for v in os.listdir(path_data):
				for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))-self.alternate * self.len_snippet, self.len_snippet):
					self.list_num_frame.append((v, i))
				self.list_num_frame.append((v, len(os.listdir(os.path.join(path_data,v,'images')))-self.len_snippet))

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		# print(self.mode)
		if self.mode == "train":
			file_name = self.video_names[idx]
			start_idx = np.random.randint(0, self.list_num_frame[idx]-self.alternate * self.len_snippet+1)
		elif self.mode == "val" or self.mode=="save":
			(file_name, start_idx) = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, file_name, 'images')
		path_annt = os.path.join(self.path_data, file_name, 'maps')

		clip_img = []
		clip_gt = []
		
		for i in range(self.len_snippet):
			img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+self.alternate*i+1))).convert('RGB')
			sz = img.size

			if self.mode!="save":
				gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+self.alternate*i+1))).convert('L'))
				gt = gt.astype('float')
				
				if self.mode == "train":
					gt = cv2.resize(gt, (384, 224))
				
				if np.max(gt) > 1.0:
					gt = gt / 255.0
				clip_gt.append(torch.FloatTensor(gt))

			clip_img.append(self.img_transform(img))
			
		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
		if self.mode!="save":
			clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))
		if self.mode=="save":
			return clip_img, start_idx, file_name, sz
		else:
			if self.multi_frame==0:
				return clip_img, clip_gt[-1]
			return clip_img, clip_gt

class Hollywood_UCFDataset(Dataset):
	def __init__(self, path_data, len_snippet, mode="train", frame_no="last", multi_frame=0):
		''' mode: train, val, perframe 
			frame_no: last, middle
		'''
		self.path_data = path_data
		self.len_snippet = len_snippet
		self.mode = mode
		self.frame_no = frame_no
		self.multi_frame = multi_frame
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
				if len(os.listdir(os.path.join(path_data,v,'images')))<=self.len_snippet:
					self.list_num_frame.append((v, 0))
		
	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		if self.mode == "train":
			file_name = self.video_names[idx]
			start_idx = np.random.randint(0, max(1, self.list_num_frame[idx]-self.len_snippet+1))
		elif self.mode == "val":
			(file_name, start_idx) = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, file_name, 'images')
		path_annt = os.path.join(self.path_data, file_name, 'maps')

		clip_img = []
		clip_gt = []

		list_clips = os.listdir(path_clip)
		list_clips.sort()
		list_sal_clips = os.listdir(path_annt)
		list_sal_clips.sort()

		if len(list_sal_clips)<self.len_snippet:
			temp = [list_clips[0] for _ in range(self.len_snippet-len(list_clips))]
			temp.extend(list_clips)
			list_clips = copy.deepcopy(temp)

			temp = [list_sal_clips[0] for _ in range(self.len_snippet-len(list_sal_clips))]
			temp.extend(list_sal_clips)
			list_sal_clips = copy.deepcopy(temp)

			assert len(list_sal_clips) == self.len_snippet and len(list_clips)==self.len_snippet

		for i in range(self.len_snippet):
			img = Image.open(os.path.join(path_clip, list_clips[start_idx+i])).convert('RGB')
			clip_img.append(self.img_transform(img))

			gt = np.array(Image.open(os.path.join(path_annt, list_sal_clips[start_idx+i])).convert('L'))
			gt = gt.astype('float')
			
			if self.mode == "train":
				gt = cv2.resize(gt, (384, 224))
			
			if np.max(gt) > 1.0:
				gt = gt / 255.0
			clip_gt.append(torch.FloatTensor(gt))
			
		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
		if self.multi_frame==0:
			gt = clip_gt[-1]
		else:
			gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))

		return clip_img, gt
			

# class DHF1KDataset(Dataset):
# 	def __init__(self, path_data, len_snippet, mode="train", frame_no="last"):
# 		''' mode: train, val, perframe 
# 			frame_no: last, middle
# 		'''
# 		self.path_data = path_data
# 		self.len_snippet = len_snippet
# 		self.mode = mode
# 		self.frame_no = frame_no
# 		print(self.frame_no)
# 		self.img_transform = transforms.Compose([
# 			transforms.Resize((224, 384)),
# 			transforms.ToTensor(),
# 			transforms.Normalize(
# 				[0.485, 0.456, 0.406],
# 				[0.229, 0.224, 0.225]
# 			)
# 		])
# 		if self.mode == "train":
# 			self.video_names = os.listdir(path_data)
# 			self.list_num_frame = [len(os.listdir(os.path.join(path_data,d,'images'))) for d in self.video_names]
# 		elif self.mode=="val":
# 			self.list_num_frame = []
# 			for v in os.listdir(path_data):
# 				for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))-self.len_snippet, self.len_snippet):
# 					self.list_num_frame.append((v, i))
# 		else:
# 			self.list_num_frame = []
# 			for v in os.listdir(path_data):
# 				for i in range(0, len(os.listdir(os.path.join(path_data,v,'images')))-self.len_snippet):
# 					self.list_num_frame.append((v, i, False))
# 				for i in range(0, len_snippet):
# 					self.list_num_frame.append((v, i+len_snippet-1, True))

# 	def __len__(self):
# 		return len(self.list_num_frame)

# 	def __getitem__(self, idx):
# 		isFlip = False
# 		# print(self.mode)
# 		if self.mode == "train":
# 			file_name = self.video_names[idx]
# 			start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)
# 		elif self.mode == "val":
# 			(file_name, start_idx) = self.list_num_frame[idx]
# 		else:
# 			(file_name, start_idx, isFlip) = self.list_num_frame[idx]

# 		path_clip = os.path.join(self.path_data, file_name, 'images')
# 		path_annt = os.path.join(self.path_data, file_name, 'maps')

# 		clip_img = []
# 		for i in range(self.len_snippet):
# 			if not isFlip:
# 				img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+i+1))).convert('RGB')
# 			else:
# 				img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx-i+1))).convert('RGB')
# 			clip_img.append(self.img_transform(img))
			
# 		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))

# 		if not isFlip:
# 			if self.frame_no=="middle":
# 				gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+(self.len_snippet)//2))).convert('L'))
# 			else:
# 				gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+self.len_snippet))).convert('L'))
# 		else:
# 			gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx-self.len_snippet+2))).convert('L'))
# 		gt = gt.astype('float')
		
# 		if self.mode == "train":
# 			gt = cv2.resize(gt, (384, 224))
		
# 		if np.max(gt) > 1.0:
# 			gt = gt / 255.0

# 		return clip_img, torch.FloatTensor(gt)


def get_audio_feature_vox(audioind, audiodata, clip_size, start_idx):
	len_snippet = clip_size
	# max_audio_Fs = 22050
	# min_video_fps = 10
	max_audio_win = 48320

	audio_feature  = torch.zeros(max_audio_win)
	# valid = {}
	# valid['audio']=0

	if audioind in audiodata:

		excerptstart = audiodata[audioind]['starts'][start_idx+1]
		if start_idx+len_snippet >= len(audiodata[audioind]['ends']):
			print("Exceeds size", audioind)
			sys.stdout.flush()
			excerptend = audiodata[audioind]['ends'][-1]
		else:
			excerptend = audiodata[audioind]['ends'][start_idx+len_snippet]	
		# try:
		# 	valid['audio'] = audiodata[audioind]['wav'][:, excerptstart:excerptend+1].shape[1]
		# except:
		# 	pass
		audio_feature_tmp = audiodata[audioind]['wav'][:, excerptstart:excerptend+1]

		if audio_feature_tmp.shape[1]<=audio_feature.shape[0]:
			audio_feature[:audio_feature_tmp.shape[1]] = audio_feature_tmp
		else:
			print("Audio Length Bigger")
			audio_feature = audio_feature_tmp[0,:].copy()
	# print(audio_feature.shape)
	audio_feature = preprocess(audio_feature.numpy()).astype(np.float32)
	assert audio_feature.shape == (512,300), audio_feature.shape
	audio_feature=np.expand_dims(audio_feature, 2)
	return transforms.ToTensor()(audio_feature)
