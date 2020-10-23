import torch
from torch import nn
import math
from model_utils import *

class PositionalEncoding(nn.Module):

	def __init__(self, feat_size, dropout=0.1, max_len=4):
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
		# print(x.shape, self.pe.shape)
		x = x + self.pe
		# return self.dropout(x)
		return x

class Transformer(nn.Module):
	def __init__(self, feat_size, hidden_size=256, nhead=4, num_encoder_layers=3, max_len=4, num_decoder_layers=-1, num_queries=4, spatial_dim=-1):
		super(Transformer, self).__init__()
		self.pos_encoder = PositionalEncoding(feat_size, max_len=max_len)
		encoder_layers = nn.TransformerEncoderLayer(feat_size, nhead, hidden_size)
		
		self.spatial_dim = spatial_dim
		if self.spatial_dim!=-1:
			transformer_encoder_spatial_layers = nn.TransformerEncoderLayer(spatial_dim, nhead, hidden_size)
			self.transformer_encoder_spatial = nn.TransformerEncoder(transformer_encoder_spatial_layers, num_encoder_layers)
		
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
		self.use_decoder = (num_decoder_layers != -1)
		
		if self.use_decoder:
			decoder_layers = nn.TransformerDecoderLayer(hidden_size, nhead, hidden_size)
			self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers, norm=nn.LayerNorm(hidden_size))
			self.tgt_pos = nn.Embedding(num_queries, hidden_size).weight
			assert self.tgt_pos.requires_grad == True

	def forward(self, embeddings, idx):
		''' embeddings: CxBxCh*H*W '''
		# print(embeddings.shape)
		batch_size = embeddings.size(1)

		if self.spatial_dim!=-1:
			embeddings = embeddings.permute((2,1,0))
			embeddings = self.transformer_encoder_spatial(embeddings)
			embeddings = embeddings.permute((2,1,0))

		x = self.pos_encoder(embeddings)
		x = self.transformer_encoder(x)
		if self.use_decoder:
			if idx!=-1:
				tgt_pos = self.tgt_pos[idx].unsqueeze(0)
				# print(tgt_pos.size())
				tgt_pos = tgt_pos.unsqueeze(1).repeat(1,batch_size,1)
			else:
				tgt_pos = self.tgt_pos.unsqueeze(1).repeat(1,batch_size,1)
			tgt = torch.zeros_like(tgt_pos)
			x = self.transformer_decoder(tgt + tgt_pos, x)
		return x

class VideoSaliencyMultiModel(nn.Module):
	def __init__(self, 
				transformer_in_channel=32, 
				use_transformer=True, 
				num_encoder_layers=3, 
				num_decoder_layers=3, 
				nhead=4, 
				multiFrame=32,
				spatial_dim=-1
			):
		super(VideoSaliencyMultiModel, self).__init__()
		
		self.use_transformer = use_transformer
		if self.use_transformer:
			self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
			self.conv_out_1x1 = nn.Conv3d(in_channels=1, out_channels=1024, kernel_size=1, stride=1, bias=True)
			self.transformer =  Transformer(
									4*7*12, 
									hidden_size=4*7*12, 
									nhead=nhead,
									num_encoder_layers=num_encoder_layers,
									num_decoder_layers=num_decoder_layers,
									max_len=transformer_in_channel,
									num_queries=32,
									spatial_dim=spatial_dim
								)

		self.backbone = BackBoneS3D()
		# for param in self.backbone.parameters():
		# 	param.requires_grad = False
		
		self.decoder = DecoderConvUp()
		# for param in self.decoder.parameters():
		# 	param.requires_grad = False

	def forward(self, x, idx):
		[y0, y1, y2, y3] = self.backbone(x)
		if self.use_transformer:
			y0 = self.conv_in_1x1(y0)
			y0 = y0.permute((1,0,2,3,4))
			# print(y0.shape)
			shape = y0.shape[2:]
			y0 = y0.flatten(2)
			y0 = self.transformer(y0, idx)
			# 32xNx4*7*12

			for i in range(y0.size(0)):
				q = y0[i]
				q = q.view((q.size(0), shape[0]//4, 4, shape[1], shape[2]))

				q = self.conv_out_1x1(q)
				# Nx1024x4xHxW
				if i==0:
					final_out = self.decoder(q, y1, y2, y3).unsqueeze(0)
				else:
					final_out = torch.cat((final_out, self.decoder(q, y1, y2, y3).unsqueeze(0)), 0)

		return final_out.permute((1,0,2,3))

class VideoSaliencyMultiModelParallel(nn.Module):
	def __init__(self, 
				transformer_in_channel=32, 
				use_transformer=True, 
				num_encoder_layers=3, 
				num_decoder_layers=3, 
				nhead=4, 
				multiFrame=32
			):
		super(VideoSaliencyMultiModelParallel, self).__init__()
		
		self.use_transformer = use_transformer
		if self.use_transformer:
			self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
			self.conv_out_1x1 = nn.Conv3d(in_channels=1, out_channels=1024, kernel_size=1, stride=1, bias=True)
			self.transformer =  Transformer(
									4*7*12, 
									hidden_size=4*7*12, 
									nhead=nhead,
									num_encoder_layers=num_encoder_layers,
									num_decoder_layers=num_decoder_layers,
									max_len=transformer_in_channel,
									num_queries=multiFrame
								)

		self.backbone = BackBoneS3D()
		self.decoder = DecoderConvUp()

	def forward(self, x, idx):
		batch_size = x.size(0)
		[y0, y1, y2, y3] = self.backbone(x)
		if self.use_transformer:
			y0 = self.conv_in_1x1(y0)
			y0 = y0.permute((1,0,2,3,4))
			# print(y0.shape)
			shape = y0.shape[2:]
			y0 = y0.flatten(2)
			y0 = self.transformer(y0, idx)
			# 32xNx4*7*12

			save_shape = y0.shape[:2]
			# print(y0.size())
			y0 = y0.view((y0.size(0)*y0.size(1), shape[0]//4, 4, shape[1], shape[2]))
			# print(y0.size())
			y0 = self.conv_out_1x1(y0)
			# print(y0.shape, y1.shape)
			final_out = self.decoder(y0, y1.repeat(y0.size(0), 1,1,1,1), y2.repeat(y0.size(0), 1,1,1,1), y3.repeat(y0.size(0), 1,1,1,1))
			# print(final_out.size())
			final_out = final_out.view((save_shape[0], save_shape[1], final_out.size(1),  final_out.size(2)))

		return final_out.permute((1,0,2,3))



class VideoSaliencyModel(nn.Module):
	def __init__(self, 
				transformer_in_channel=32,
				nhead=4,
				use_upsample=True,
				num_hier=3,
				num_clips=32
			):
		super(VideoSaliencyModel, self).__init__()

		self.backbone = BackBoneS3D()
		self.num_hier = num_hier
		if use_upsample:
			if num_hier==0:
				self.decoder = DecoderConvUpNoHier()
			elif num_hier==1:
				self.decoder = DecoderConvUp1Hier()
			elif num_hier==2:
				self.decoder = DecoderConvUp2Hier()
			elif num_hier==3:
				if num_clips==8:
					self.decoder = DecoderConvUp8()
				elif num_clips==16:
					self.decoder = DecoderConvUp16()
				elif num_clips==32:
					self.decoder = DecoderConvUp()
				elif num_clips==48:
					self.decoder = DecoderConvUp48()
		else:
			self.decoder = DecoderConvT()

	def forward(self, x):
		[y0, y1, y2, y3] = self.backbone(x)
		if self.num_hier==0:
			return self.decoder(y0)
		if self.num_hier==1:
			return self.decoder(y0, y1)
		if self.num_hier==2:
			return self.decoder(y0, y1, y2)
		if self.num_hier==3:
			return self.decoder(y0, y1, y2, y3)

class DecoderConvUp(nn.Module):
	def __init__(self):
		super(DecoderConvUp, self).__init__()
		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 112 x 192

			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 224 x 384

			# 4 time dimension
			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(),            
			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1, y2, y3):
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)
		
		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)
		
		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)
		
		z = self.convtsp4(z)
		# print('convtsp4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z

class DecoderConvUp16(nn.Module):
	def __init__(self):
		super(DecoderConvUp16, self).__init__()
		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 112 x 192

			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 224 x 384

			# 4 time dimension
			nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(1,1,1), bias=True),
			# nn.ReLU(),            
			# nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1, y2, y3):
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)
		
		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)
		
		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)
		
		z = self.convtsp4(z)
		# print('convtsp4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z

class DecoderConvUp8(nn.Module):
	def __init__(self):
		super(DecoderConvUp8, self).__init__()
		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 112 x 192

			nn.Conv3d(64, 32, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 224 x 384

			# 4 time dimension
			nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(1,1,1), bias=True),
			# nn.ReLU(),            
			# nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1, y2, y3):
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)
		
		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)
		
		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)
		
		z = self.convtsp4(z)
		# print('convtsp4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z

class DecoderConvUp48(nn.Module):
	def __init__(self):
		super(DecoderConvUp48, self).__init__()
		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 112 x 192

			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 224 x 384

			# 4 time dimension
			nn.Conv3d(32, 32, kernel_size=(3,1,1), stride=(3,1,1), bias=True),
			nn.ReLU(),            
			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1, y2, y3):
		# print(y0.shape)
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)
		
		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)
		
		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)
		
		z = self.convtsp4(z)
		# print('convtsp4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z


class DecoderConvUpNoHier(nn.Module):
	def __init__(self):
		super(DecoderConvUpNoHier, self).__init__()
		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 112 x 192

			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 224 x 384

			# 4 time dimension
			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(),            
			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0):
		
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape)

		# z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)
		
		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		# z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)
		
		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		# z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)
		
		z = self.convtsp4(z)
		# print('convtsp4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z

class DecoderConvUp1Hier(nn.Module):
	def __init__(self):
		super(DecoderConvUp1Hier, self).__init__()
		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 112 x 192

			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 224 x 384

			# 4 time dimension
			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(),            
			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1):
		
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape, y1.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)
		
		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		# z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)
		
		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		# z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)
		
		z = self.convtsp4(z)
		# print('convtsp4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z

class DecoderConvUp2Hier(nn.Module):
	def __init__(self):
		super(DecoderConvUp2Hier, self).__init__()
		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 112 x 192

			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 224 x 384

			# 4 time dimension
			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(),            
			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1, y2):
		
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)
		
		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)
		
		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		# z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)
		
		z = self.convtsp4(z)
		# print('convtsp4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z

class BackBoneS3D(nn.Module):
	def __init__(self):
		super(BackBoneS3D, self).__init__()
		
		self.base1 = nn.Sequential(
			SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
			BasicConv3d(64, 64, kernel_size=1, stride=1),
			SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
		)
		self.maxp2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
		self.base2 = nn.Sequential(
			Mixed_3b(),
			Mixed_3c(),
		)
		self.maxp3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
		self.base3 = nn.Sequential(
			Mixed_4b(),
			Mixed_4c(),
			Mixed_4d(),
			Mixed_4e(),
			Mixed_4f(),
		)
		self.maxt4 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0))
		self.maxp4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
		self.base4 = nn.Sequential(
			Mixed_5b(),
			Mixed_5c(),
		)

	def forward(self, x):
		# print('input', x.shape)
		y3 = self.base1(x)
		# print('base1', y3.shape)
		
		y = self.maxp2(y3)
		# print('maxp2', y.shape)

		y2 = self.base2(y)
		# print('base2', y2.shape)

		y = self.maxp3(y2)
		# print('maxp3', y.shape)

		y1 = self.base3(y)
		# print('base3', y1.shape)

		y = self.maxt4(y1)
		y = self.maxp4(y)
		# print('maxt4p4', y.shape)

		y0 = self.base4(y)

		return [y0, y1, y2, y3]


# class DecoderConvUp8Frame(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvUp8Frame, self).__init__()
# 		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(5,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),            
# 			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2, y3):
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(2), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z

class DecoderConvT(nn.Module):
	def __init__(self):
		super(DecoderConvT, self).__init__()
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 1024, kernel_size=1, stride=1, bias=False),
			nn.ReLU(),

			nn.ConvTranspose3d(1024, 832, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
			nn.ReLU(),
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 832, kernel_size=(3,1,1), stride=(3,1,1), bias=False),
			nn.ReLU(),

			nn.ConvTranspose3d(832, 480, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
			nn.ReLU(),
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 480, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
			nn.ReLU(),

			nn.ConvTranspose3d(480, 192, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
			nn.ReLU(),
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 192, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
			nn.ReLU(),

			nn.ConvTranspose3d(192, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
			nn.ReLU(),

			nn.Conv3d(64, 64, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(),

			nn.ConvTranspose3d(64, 4, kernel_size=1, stride=1, bias=False),
			nn.ReLU(),

			nn.Conv3d(4, 4, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(),

			nn.ConvTranspose3d(4, 4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
			nn.Conv3d(4, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1, y2, y3):
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)
		
		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)
		
		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)
		
		z = self.convtsp4(z)
		# print('convtsp4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z

# class DecoderConvTDualFrame(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvTDualFrame, self).__init__()

# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 1024, kernel_size=1, stride=1, bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(1024, 832, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 832, kernel_size=(3,1,1), stride=(3,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(832, 480, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 480, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(480, 192, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 		)

# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 192, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(192, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.Conv3d(64, 64, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(64, 4, kernel_size=1, stride=1, bias=False),
# 			nn.ReLU(),

# 			nn.Conv3d(4, 4, kernel_size=(1,1,1), stride=(1,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(4, 4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.Conv3d(4, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2, y3):
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(2), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z

# class DecoderConvTMulti(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvTMulti, self).__init__()

# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 1024, kernel_size=1, stride=1, bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(1024, 832, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 832, kernel_size=(3,1,1), stride=(3,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(832, 480, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 480, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(480, 192, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.ConvTranspose3d(192, 64, kernel_size=(5,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.Conv3d(64, 64, kernel_size=(3,1,1), padding=(1,0,0), stride=(1,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(64, 4, kernel_size=(5,1,1), stride=1, bias=False),
# 			nn.ReLU(),

# 			nn.Conv3d(4, 4, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(4, 4, kernel_size=(5,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.Conv3d(4, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2, y3):
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(2), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z

# class TASED_v2_hier(nn.Module):
# 	def __init__(self, transformer_in_channel=32, use_transformer=True, num_encoder_layers=3, nhead=4):
# 		super(TASED_v2_hier, self).__init__()
# 		self.use_transformer = use_transformer
# 		if self.use_transformer:
# 			self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
# 			self.conv_out_1x1 = nn.Conv3d(in_channels=transformer_in_channel, out_channels=1024, kernel_size=1, stride=1, bias=True)
# 			self.transformer = Transformer(transformer_in_channel*7*12, hidden_size=transformer_in_channel*7*12, nhead=nhead, num_encoder_layers=num_encoder_layers)
# 		self.base1 = nn.Sequential(
# 			SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
# 			nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
# 			BasicConv3d(64, 64, kernel_size=1, stride=1),
# 			SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
# 		)
# 		self.maxp2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
# 		self.base2 = nn.Sequential(
# 			Mixed_3b(),
# 			Mixed_3c(),
# 		)
# 		self.maxp3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
# 		self.base3 = nn.Sequential(
# 			Mixed_4b(),
# 			Mixed_4c(),
# 			Mixed_4d(),
# 			Mixed_4e(),
# 			Mixed_4f(),
# 		)
# 		self.maxt4 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0))
# 		self.maxp4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
# 		self.base4 = nn.Sequential(
# 			Mixed_5b(),
# 			Mixed_5c(),
# 		)

# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 1024, kernel_size=1, stride=1, bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(1024, 832, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 832, kernel_size=(3,1,1), stride=(3,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(832, 480, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 480, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(480, 192, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 192, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(192, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.Conv3d(64, 64, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(64, 4, kernel_size=1, stride=1, bias=False),
# 			nn.ReLU(),

# 			nn.Conv3d(4, 4, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),

# 			nn.ConvTranspose3d(4, 4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
# 			nn.Conv3d(4, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

		

# 	def forward(self, x):
# 		# print('input', x.shape)
# 		y3 = self.base1(x)
# 		# print('base1', y3.shape)
		
# 		y = self.maxp2(y3)
# 		# print('maxp2', y.shape)

# 		y2 = self.base2(y)
# 		# print('base2', y2.shape)

# 		y = self.maxp3(y2)
# 		# print('maxp3', y.shape)

# 		y1 = self.base3(y)
# 		# print('base3', y1.shape)

# 		y = self.maxt4(y1)
# 		y = self.maxp4(y)
# 		# print('maxt4p4', y.shape)

# 		y0 = self.base4(y)
# 		# NxCxTxWxH
# 		# print(y0.shape)
		
# 		if self.use_transformer:
# 			y0 = self.conv_in_1x1(y0)
# 			y0 = y0.permute((2,0,1,3,4))
# 			# print(y0.shape)
# 			shape = y0.size()
# 			y0 = y0.flatten(2)

# 			y0 = self.transformer(y0)
# 			# 4xNx832x7x12

# 			y0 = y0.view(shape)
# 			y0 = y0.permute((1,2,0,3,4))
# 			y0 = self.conv_out_1x1(y0)
# 		# print('base4', y.shape)


# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
# 		z = z.view(z.size(0), z.size(3), z.size(4))        
# 		# print('output', z.shape)

# 		return z

# class DecoderConvUpChannelConcat(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvUpChannelConcat, self).__init__()
# 		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')

# 		self.conv_hier1 = nn.Conv3d(832, 832, kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0), bias=False)
# 		self.conv_hier2 = nn.Conv3d(480, 480, kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0), bias=False)
# 		self.conv_hier3 = nn.Conv3d(192, 192, kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0), bias=False)

# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			# nn.Upsample(scale_factor=(2,2,2), mode='trilinear')
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832+832, 480, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			# nn.Upsample(scale_factor=(8,2,2), mode='trilinear')
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480+480, 192, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192+192, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),            
# 			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2, y3):
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape, y1.shape)

# 		y1 = self.conv_hier1(y1)
# 		z = torch.cat((z,y1), 1)
# 		# print('cat_convtsp1', z.shape)
# 		# print(z.shape)
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape, y2.shape)

# 		y2 = self.conv_hier2(y2)
# 		# print(z.shape, y2.shape)
# 		z = torch.cat((z,y2), 1)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape, y3.shape)

# 		y3 = self.conv_hier3(y3)
# 		z = torch.cat((z,y3), 1)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
# 		# print(z.shape)
# 		z = z.view(z.size(0), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z

# class VideoSaliencyChannelConcat(nn.Module):
# 	def __init__(self, 
# 				transformer_in_channel=32,
# 				use_transformer=True,  
# 				num_encoder_layers=3, 
# 				num_decoder_layers=-1, 
# 				nhead=4, 
# 				multiFrame=0
# 			):
# 		super(VideoSaliencyChannelConcat, self).__init__()
		
# 		self.backbone = BackBoneS3D()    
# 		self.use_transformer = use_transformer
# 		if self.use_transformer:
# 			self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
# 			self.conv_out_1x1 = nn.Conv3d(in_channels=transformer_in_channel, out_channels=1024, kernel_size=1, stride=1, bias=True)
# 			self.transformer =  Transformer(
# 									4*7*12, 
# 									hidden_size=4*7*12, 
# 									nhead=nhead,
# 									num_encoder_layers=num_encoder_layers,
# 									num_decoder_layers=num_decoder_layers,
# 									max_len=transformer_in_channel
# 								)
# 		self.decoder = DecoderConvUpChannelConcat()

# 	def forward(self, x):
# 		[y0, y1, y2, y3] = self.backbone(x)
# 		if self.use_transformer:
# 			# print("Inside Transformer")
# 			y0 = self.conv_in_1x1(y0)
# 			# BxChxClxHxW
# 			y0 = y0.permute((1,0,2,3,4))
# 			# print(y0.shape)
# 			shape = y0.size()
# 			y0 = y0.flatten(2)

# 			y0 = self.transformer(y0)
# 			# 32xNx4*7*12

# 			y0 = y0.view(shape)
# 			y0 = y0.permute((1,0,2,3,4))
# 			# print(y0.shape)
# 			y0 = self.conv_out_1x1(y0)
# 		return self.decoder(y0, y1, y2, y3)

# class VideoSaliencyChannel(nn.Module):
# 	def __init__(self, 
# 				transformer_in_channel=32, 
# 				use_transformer=True, 
# 				num_encoder_layers=3, 
# 				num_decoder_layers=-1, 
# 				nhead=4, 
# 				multiFrame=0,
# 				use_upsample=True
# 			):
# 		super(VideoSaliencyChannel, self).__init__()
		
# 		self.use_transformer = use_transformer
# 		if self.use_transformer:
# 			self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
# 			self.conv_out_1x1 = nn.Conv3d(in_channels=transformer_in_channel, out_channels=1024, kernel_size=1, stride=1, bias=True)
# 			self.transformer =  Transformer(
# 									4*7*12, 
# 									hidden_size=4*7*12, 
# 									nhead=nhead,
# 									num_encoder_layers=num_encoder_layers,
# 									num_decoder_layers=num_decoder_layers,
# 									max_len=transformer_in_channel
# 								)

# 		self.backbone = BackBoneS3D()    
# 		if use_upsample:
# 			self.decoder = DecoderConvUp()
# 		else:
# 			self.decoder = DecoderConvT()

# 	def forward(self, x):
# 		[y0, y1, y2, y3] = self.backbone(x)
# 		if self.use_transformer:
# 			# print("Inside Transformer")
# 			y0 = self.conv_in_1x1(y0)
# 			# BxChxClxHxW
# 			y0 = y0.permute((1,0,2,3,4))
# 			# print(y0.shape)
# 			shape = y0.size()
# 			y0 = y0.flatten(2)

# 			y0 = self.transformer(y0)
# 			# 32xNx4*7*12

# 			y0 = y0.view(shape)
# 			y0 = y0.permute((1,0,2,3,4))
# 			# print(y0.shape)
# 			y0 = self.conv_out_1x1(y0)
		
# 		return self.decoder(y0, y1, y2, y3)