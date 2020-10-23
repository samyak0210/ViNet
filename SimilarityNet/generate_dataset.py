import os
import sys
from os.path import join
import json
from loss import similarity
import cv2

threshold = 0.95

path = '/ssd_scratch/cvit/samyak/Results/DHF1K_training_set'
json_file = 'training_set.json'

data = {}

for video in sorted(os.listdir(path)):
	print(video)
	sys.stdout.flush()
	data[video] = []
	frame_names = sorted(os.listdir(join(path, video)))
	for i, anchor in enumerate(frame_names[:450]):
		positive, negative = None, None

		idx = i+1
		anchor_map = cv2.imread(join(path, video, anchor), 0)
		while idx<len(frame_names):
			sample = cv2.imread(join(path, video, frame_names[idx]), 0)
			# print(similarity(anchor_map, sample))
			if similarity(anchor_map, sample) < threshold:
				negative = frame_names[idx]
				break
			idx+=1
		
		if negative!=None and idx-i>=2:
			_dic = {
				'anchor': anchor,
				'positive': frame_names[(i+idx)//2],
				'negative': negative
			}
			data[video].append(_dic)


with open(json_file, 'w') as f:
	json.dump(data, f, sort_keys=True, indent=4)