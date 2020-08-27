from model_hier import *

a = VideoSaliencyChannelConcat(use_transformer=False).cuda()

b = torch.zeros((1,3,32,224,384)).cuda()
with torch.no_grad():
	print(a(b).shape)