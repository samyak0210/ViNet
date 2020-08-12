import torch
from torch import nn
import math
from model_utils import *
from efficientnet_pytorch import EfficientNet

class BackBoneENetB3(nn.Module):
	def __init__(self, finetune=True, get_multilevel_features=True):
		super(BackBoneENetB3, self).__init__()

		self.model = EfficientNet.from_pretrained('efficientnet-b3')
		for (name, param) in self.model.named_parameters():
			if "_fc"==name.split('.')[0]:
				param.requires_grad = False
			else:
				param.requires_grad = finetune

		self.conv_block1 = nn.Sequential(
			self.model._conv_stem,
			self.model._bn0,
			self.model._blocks[0],
			self.model._blocks[1] 
		)

		self.conv_block5 = nn.Sequential(
			self.model._conv_head,
			self.model._bn1
		)
		self.get_multilevel_features = get_multilevel_features
		self.out_channels = self.model._bn1.num_features

	def forward(self, image):
		batch_size = images.size(0)

		out = self.conv_block1(images)
		out1 = out

		for i in range(2, len(self.model._blocks)):
			out = self.model._blocks[i](out)
			if i==4:
				out2 = out
			if i==7:
				out3 = out 
			if i==17:
				out4 = out
			if i==len(self.model._blocks)-1:
				out5 = out

		out5 = self.conv_block5(out5)
		
		assert out1.size() == (batch_size, 24, 128, 128)
		assert out2.size() == (batch_size, 32, 64, 64)
		assert out3.size() == (batch_size, 48, 32, 32)
		assert out4.size() == (batch_size, 136, 16, 16)
		assert out5.size() == (batch_size, 1536, 8, 8)

		if self.get_multilevel_features:
			return (out1, out2, out3, out4, out5)
		else:
			return out5

class Decoder(nn.Module):
	def __init__(self, channels_list, use_multilevel_features=True):
		super(Decoder, self).__init__()
		self.use_multilevel_features = use_multilevel_features
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

	def forward(self, inp, out=None):
		''' inp: BxChxHxW '''
		x = self.deconv_layer0(inp)

		if not out is None: 
			[out1, out2, out3, out4] = out
		
		if self.use_multilevel_features:
			x = torch.cat((x,out4), 1)
		x = self.deconv_layer1(x)
		
		if self.use_multilevel_features:
			x = torch.cat((x,out3), 1)
		x = self.deconv_layer2(x)

		if self.use_multilevel_features:
			x = torch.cat((x,out2), 1)
		x = self.deconv_layer3(x)
		
		if self.use_multilevel_features:
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


class CombinedModel(nn.Module):
	def __init__(self,  
		hidden_channel, 
		kernel_size=(3,3), 
		bias=True,
		finetune=True,
		get_multilevel_features=True
	):
		self.get_multilevel_features = get_multilevel_features
		self.backbone = BackBoneENetB3(finetune=finetune, get_multilevel_features=get_multilevel_features)
		self.ConvLSTMCell = ConvLSTMCell(self.backbone.out_channels, hidden_channel, lstm_out_channels, kernel_size=kernel_size, bias=bias)
		if get_multilevel_features:
			print("Using MultiLevel Features")
			channels_list = [
				[hidden_channel, hidden_channel//2],
				[hidden_channel//2 + 136, hidden_channel//4],
				[hidden_channel//4 + 48, hidden_channel//8],
				[hidden_channel//8 + 32, hidden_channel//16],
				[hidden_channel//16 + 24, hidden_channel//32],
				[hidden_channel//32, hidden_channel//32]
			]
		else:
			channels_list = [
				[hidden_channel, hidden_channel//2],
				[hidden_channel//2, hidden_channel//4],
				[hidden_channel//4, hidden_channel//8],
				[hidden_channel//8, hidden_channel//16],
				[hidden_channel//16, hidden_channel//32],
				[hidden_channel//32, hidden_channel//32]
			]

		self.decoder = Decoder(channels_list, use_multilevel_features=get_multilevel_features)

	def forward(self, x):
		''' x: ClxBxCxHxW '''
		clip_size, batch_size, _, H, W = x.size()
		
		(h, c) = self.ConvLSTMCell.init_hidden(batch_size, image_size=(H,W))

		final_out = None
		for idx, t in enumerate(range(clip_size)):
			if get_multilevel_features:
				(out1, out2, out3, out4, out5) = self.backbone(x[t, :, :, :, :])
			else:
				out5 = self.backbone(x[t, :, :, :, :])

			h, c = self.ConvLSTMCell(out5, cur_state=[h,c])

			if get_multilevel_features:
				out = self.decoder(out5, [out1, out2, out3, out4])
			else:
				out = self.decoder(out5)

			if idx==0:
				final_out = out.unsqueeze(0)
			else:
				final_out = torch.cat((final_out, out.unsqueeze(0)), 0)

		assert final_out.size() == (clip_size, H, W)
		return final_out