import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../saliency/PNAS/')
from PNASnet import *
from genotypes import PNASNet
from efficientnet_pytorch import EfficientNet
import math

class BackBonePNAS(nn.Module):
	def __init__(self, num_channels=3, finetune=True, load_weight=1):
		super(BackBonePNAS, self).__init__()
		self.path = '../saliency/PNAS/PNASNet-5_Large.pth'

		self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)
		if load_weight:
			self.pnas.load_state_dict(torch.load(self.path))
		
		for param in self.pnas.parameters():
			param.requires_grad = finetune

		self.padding = nn.ConstantPad2d((0,1,0,1),0)
		self.drop_path_prob = 0

	def forward(self, images):
		batch_size = images.size(0)

		s0 = self.pnas.conv0(images)
		s0 = self.pnas.conv0_bn(s0)
		out1 = self.padding(s0)

		s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
		out2 = s1
		s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

		for i, cell in enumerate(self.pnas.cells):
			s0, s1 = s1, cell(s0, s1, 0)
			if i==3:
				out3 = s1
			if i==7:
				out4 = s1
			if i==11:
				out5 = s1

		return [out1, out2, out3, out4], out5

class BackBoneENet(nn.Module):
	def __init__(self, num_channels=3, finetune=True, load_weight=1):
		super(BackBoneENet, self).__init__()

		self.model = EfficientNet.from_pretrained('efficientnet-b0')
		for param in self.model.parameters():
			param.requires_grad = finetune

		self.conv_block1 = nn.Sequential(
			self.model._conv_stem,
			self.model._bn0,
			self.model._blocks[0]
		)

		self.conv_block5 = nn.Sequential(
			self.model._conv_head,
			self.model._bn1
		)

	def forward(self, images):
		batch_size = images.size(0)

		# out = self.conv_block1(images)
		# out1 = out
		# for i in range(1, len(self.model._blocks)):
		# 	out = self.model._blocks[i](out)
		# 	if i==2:
		# 		out2 = out
		# 	if i==4:
		# 		out3 = out 
		# 	if i==10:
		# 		out4 = out
		# 	if i==len(self.model._blocks)-1:
		# 		out5 = out

		# out5 = self.conv_block5(out5)
		
		out5 = self.model.extract_features(images)
		# assert out1.size() == (batch_size, 16, 128, 128)
		# assert out2.size() == (batch_size, 24, 64, 64)
		# assert out3.size() == (batch_size, 40, 32, 32)
		# assert out4.size() == (batch_size, 112, 16, 16)
		# assert out5.size() == (batch_size, 1280, 8, 8)

		return out5

class PositionalEncoding(nn.Module):

	def __init__(self, feat_size, dropout=0.1, max_len=32):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, feat_size)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, feat_size, 2).float() * (-math.log(10000.0) / feat_size))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe
		# return self.dropout(x)
		return x

class Transformer(nn.Module):
	def __init__(self, feat_size, hidden_size=256, nhead=8, num_encoder_layers=6, max_len=32):
		super(Transformer, self).__init__()
		self.pos_encoder = PositionalEncoding(feat_size, max_len=max_len)
		encoder_layers = nn.TransformerEncoderLayer(feat_size, nhead, hidden_size)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)


	def forward(self, embeddings):
		''' embeddings: CxBxCh*H*W '''
		x = self.pos_encoder(embeddings)
		x = self.transformer_encoder(x)
		return x

class Decoder(nn.Module):
	def __init__(self, channels_list):
		super(Decoder, self).__init__()
		self.deconv_layer0 = self.decoderBlock(channels_list[0][0], channels_list[0][1])
		self.deconv_layer1 = self.decoderBlock(channels_list[1][0], channels_list[1][1])
		self.deconv_layer2 = self.decoderBlock(channels_list[2][0], channels_list[2][1])
		self.deconv_layer3 = self.decoderBlock(channels_list[3][0], channels_list[3][1])
		self.deconv_layer4 = self.decoderBlock(channels_list[4][0], channels_list[4][1])
		self.deconv_layer5 = nn.Sequential(
								nn.Conv2d(in_channels = channels_list[5][0], out_channels = channels_list[5][1], kernel_size = 3, padding = 1, bias = True),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels = channels_list[5][1], out_channels = 1, kernel_size = 3, padding = 1, bias = True),
								nn.Sigmoid()
							)

	def forward(self, inp, out):
		''' inp: BxChxHxW '''
		x = self.deconv_layer0(inp)

		[out1, out2, out3, out4] = out
		x = torch.cat((x,out4), 1)
		x = self.deconv_layer1(x)
		
		x = torch.cat((x,out3), 1)
		x = self.deconv_layer2(x)

		x = torch.cat((x,out2), 1)
		x = self.deconv_layer3(x)
		
		x = torch.cat((x,out1), 1)
		x = self.deconv_layer4(x)
		x = self.deconv_layer5(x)
		x = x.squeeze(1)
		
		return x

	def decoderBlock(self, in_channels, out_channels, kernel_size=3, padding=1, bias = True):
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
			nn.ReLU(inplace=True),
			nn.UpsamplingBilinear2d(scale_factor=2)
		)


class DecoderWithoutSkip(nn.Module):
	def __init__(self, channels_list):
		super(DecoderWithoutSkip, self).__init__()
		self.deconv_layer0 = self.decoderBlock(channels_list[0][0], channels_list[0][1])
		self.deconv_layer1 = self.decoderBlock(channels_list[1][0], channels_list[1][1])
		self.deconv_layer2 = self.decoderBlock(channels_list[2][0], channels_list[2][1])
		self.deconv_layer3 = self.decoderBlock(channels_list[3][0], channels_list[3][1])
		self.deconv_layer4 = self.decoderBlock(channels_list[4][0], channels_list[4][1])
		self.deconv_layer5 = nn.Sequential(
								nn.Conv2d(in_channels = channels_list[5][0], out_channels = channels_list[5][1], kernel_size = 3, padding = 1, bias = True),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels = channels_list[5][1], out_channels = 1, kernel_size = 3, padding = 1, bias = True),
								nn.Sigmoid()
							)

	def forward(self, inp):
		''' inp: BxChxHxW '''
		x = self.deconv_layer0(inp)
		x = self.deconv_layer1(x)		
		x = self.deconv_layer2(x)
		x = self.deconv_layer3(x)
		x = self.deconv_layer4(x)
		x = self.deconv_layer5(x)
		x = x.squeeze(1)
		
		return x

	def decoderBlock(self, in_channels, out_channels, kernel_size=3, padding=1, bias = True):
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
			nn.ReLU(inplace=True),
			nn.UpsamplingBilinear2d(scale_factor=2)
		)

class VideoSaliencyModel(nn.Module):
	def __init__(self, transformer_in_channel=1280, transformer_out_channel=64, finetune=True, load_weight=1, nhead=8, num_encoder_layers=6, clip_size=32):
		super(VideoSaliencyModel, self).__init__()
		self.backbone = BackBoneENet(finetune=finetune, load_weight=load_weight)
		self.transformer_out_channel = transformer_out_channel
		self.conv = nn.Conv2d(in_channels=transformer_in_channel, out_channels=transformer_out_channel, kernel_size=(1,1), bias=True)
		self.transformer = Transformer(transformer_out_channel * 8 * 8, hidden_size=transformer_out_channel * 8 * 8, nhead=nhead, num_encoder_layers=num_encoder_layers, max_len=clip_size)
		channels_list = [
			(transformer_out_channel*clip_size, 256),
			(256, 128),
			(128, 64),
			(64, 32),
			(32, 16),
			(16, 16)
		]
		self.decoder = DecoderWithoutSkip(channels_list)

	def forward(self, clips):
		''' clips: BxCxChxHxW '''
		batch_size = clips.size(0)
		num_clips = clips.size(1)

		clips = clips.permute((1,0,2,3,4))
		# outputs = []
		for i in range(num_clips):
			final_out = self.backbone(clips[i])
			if i==0:
				features = self.conv(final_out).unsqueeze(0)
			else:
				features = torch.cat((features, self.conv(final_out).unsqueeze(0)), 0)
			# outputs.append(out)

		features = features.flatten(2)
		features = self.transformer(features)
		assert features.size(0) == num_clips and features.size(1) == batch_size 
		
		features = features.permute((1,0,2))
		features = features.reshape(batch_size, num_clips*self.transformer_out_channel, 8, 8)
		# for i in range(num_clips):
		# 	if i==0:
		# 		sal_map = self.decoder(features[i]).unsqueeze(0)
		# 	else:
		# 		sal_map = torch.cat((sal_map, self.decoder(features[i]).unsqueeze(0)), 0)
		sal_map = self.decoder(features)
		# sal_map = sal_map.permute((1,0,2,3))
		return sal_map

model = torch.nn.DataParallel(VideoSaliencyModel(nhead=4, num_encoder_layers=3, finetune=False)).cuda()
a = torch.zeros((16,32,3,256,256)).cuda()
b=model(a)
print(b.size())
