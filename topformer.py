from backbone import Backbone
import torch, math
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F

class Head(nn.Module):
	def __init__(self, c=256):
		super().__init__()
		self.conv_seg = nn.Conv2d(c, 2, kernel_size=1)
		self.linear_fuse = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(c, c, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)),
			('bn', nn.BatchNorm2d(c))
		]))
		self.act = nn.ReLU6()
		self.dropout = nn.Dropout2d(0.1)
	def forward(self, x):
		x=self.act(self.linear_fuse(x))
		x=self.dropout(x)
		return self.conv_seg(x)

class TopFormer(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.backbone=Backbone()
		self.decode_head=Head()	
		self.init_weights()
		
	def forward(self, x):
		B, C, H, W = x.shape
		x=self.backbone(x)
		xx=x[0]
		for i in x[1:]:
			xx += F.interpolate(i, xx.size()[2:], mode='bilinear', align_corners=False)
		xx = self.decode_head(xx)
		return F.interpolate(xx, (H,W), mode='bilinear', align_corners=False)
		
	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				n //= m.groups
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None: m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None: m.bias.data.zero_()
		
if __name__ == '__main__':
	topf=TopFormer().cuda()
	data=torch.rand((8,3,256,256)).cuda()
	for _ in range(1000):
		res=topf(data)
	print(res.shape)
