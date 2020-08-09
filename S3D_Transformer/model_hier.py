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
        x = x + self.pe
        # return self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self, feat_size, hidden_size=256, nhead=8, num_encoder_layers=6, max_len=4):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feat_size, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(feat_size, nhead, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)


    def forward(self, embeddings):
        ''' embeddings: CxBxCh*H*W '''
        # print(embeddings.shape)
        x = self.pos_encoder(embeddings)
        x = self.transformer_encoder(x)
        return x

class VideoSaliencyModel(nn.Module):
    def __init__(self, transformer_in_channel=32, use_transformer=False, num_encoder_layers=3, nhead=4):
        super(VideoSaliencyModel, self).__init__()
        
        self.use_transformer = use_transformer
        if self.use_transformer:
            self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
            self.conv_out_1x1 = nn.Conv3d(in_channels=transformer_in_channel, out_channels=1024, kernel_size=1, stride=1, bias=True)
            self.transformer = Transformer(transformer_in_channel*7*12, hidden_size=transformer_in_channel*7*12, nhead=nhead, num_encoder_layers=num_encoder_layers)

        self.backbone = BackBoneS3D()
        self.decoder = DecoderConvT()

    def forward(self, x):
        [y0, y1, y2, y3] = self.backbone(x)
        if self.use_transformer:
            y0 = self.conv_in_1x1(y0)
            y0 = y0.permute((2,0,1,3,4))
            # print(y0.shape)
            shape = y0.size()
            y0 = y0.flatten(2)

            y0 = self.transformer(y0)
            # 4xNx832x7x12

            y0 = y0.view(shape)
            y0 = y0.permute((1,2,0,3,4))
            y0 = self.conv_out_1x1(y0)
        
        return self.decoder(y0, y1, y2, y3)

class DecoderConvUpsample(nn.Module):
    def __init__(self):
        super(DecoderConvUpsample, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')

class DecoderConvT(nn.Module):
    def __init__(self):
        super(DecoderConvT, self).__init__()

        self.convtsp1 = nn.Sequential(
            nn.Conv3d(1024, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(1024, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(1024, 832, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(832, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(832, 832, kernel_size=(3,1,1), stride=(3,1,1), bias=False),
            nn.BatchNorm3d(832, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(832, 480, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(480, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(480, 480, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
            nn.BatchNorm3d(480, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(480, 192, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(192, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
            nn.BatchNorm3d(192, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(192, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(64, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.Conv3d(64, 64, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
            nn.BatchNorm3d(64, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(4, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.Conv3d(4, 4, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
            nn.BatchNorm3d(4, eps=1e-3, momentum=0.001, affine=True),
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


class TASED_v2_hier(nn.Module):
    def __init__(self, transformer_in_channel=32, use_transformer=False, num_encoder_layers=3, nhead=4):
        super(TASED_v2_hier, self).__init__()
        self.use_transformer = use_transformer
        if self.use_transformer:
            self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
            self.conv_out_1x1 = nn.Conv3d(in_channels=transformer_in_channel, out_channels=1024, kernel_size=1, stride=1, bias=True)
            self.transformer = Transformer(transformer_in_channel*7*12, hidden_size=transformer_in_channel*7*12, nhead=nhead, num_encoder_layers=num_encoder_layers)
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

        self.convtsp1 = nn.Sequential(
            nn.Conv3d(1024, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(1024, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(1024, 832, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(832, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(832, 832, kernel_size=(3,1,1), stride=(3,1,1), bias=False),
            nn.BatchNorm3d(832, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(832, 480, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(480, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(480, 480, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
            nn.BatchNorm3d(480, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(480, 192, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(192, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(5,1,1), stride=(5,1,1), bias=False),
            nn.BatchNorm3d(192, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(192, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(64, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.Conv3d(64, 64, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
            nn.BatchNorm3d(64, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(4, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.Conv3d(4, 4, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
            nn.BatchNorm3d(4, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(4, 4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.Conv3d(4, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
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
        # NxCxTxWxH
        # print(y0.shape)
        
        if self.use_transformer:
            y0 = self.conv_in_1x1(y0)
            y0 = y0.permute((2,0,1,3,4))
            # print(y0.shape)
            shape = y0.size()
            y0 = y0.flatten(2)

            y0 = self.transformer(y0)
            # 4xNx832x7x12

            y0 = y0.view(shape)
            y0 = y0.permute((1,2,0,3,4))
            y0 = self.conv_out_1x1(y0)
        # print('base4', y.shape)


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