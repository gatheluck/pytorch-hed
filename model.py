# import libraries
import re
import copy
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def weights_init(m):
	classname = m.__class__.__name__
	print (classname)
	if classname.find('Conv2d') != -1:
		init.xavier_uniform(m.weight.data)
		init.constant(m.bias.data, 0.1)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def transfer_weights(model_from, model_to):
	wf = copy.deepcopy(model_from.state_dict())
	wt = model_to.state_dict()
	for k in wt.keys():
		if not k in wf:
			wf[k] = wt[k]
	model_to.load_state_dict(wf)

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
				identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
				identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class custom_resnet(nn.Module):
	def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
		super(custom_resnet, self).__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		# x = self.avgpool(x)
		# x = x.view(x.size(0), -1)
		# x = self.fc(x)

		return x

class custom_vgg16(nn.Module):
	def __init__(self):
		super(custom_vgg16, self).__init__()

		# conv1
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=35),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		# conv2
		self.conv2 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		# conv3
		self.conv3 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
			nn.Conv2d(128, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		# conv4
		self.conv4 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
			nn.Conv2d(256, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		# conv5
		self.conv5 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)
		return conv5

class custom_vgg16_bn(nn.Module):
	def __init__(self):
		super(custom_vgg16_bn, self).__init__()

		# conv1
		self.conv1 = nn.Sequential(
			# 3x64 
			nn.Conv2d(3, 64, 3, padding=35),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			# 64x64
			nn.Conv2d(64, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)

		# conv2
		self.conv2 = nn.Sequential(
			# max pooling
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
			# 128x128
			nn.Conv2d(64, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			# 128x128
			nn.Conv2d(128, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)

		# conv3
		self.conv3 = nn.Sequential(
			# max pooling
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
			# 256x256
			nn.Conv2d(128, 256, 3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			# 256x256
			nn.Conv2d(256, 256, 3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			# 256x256
			nn.Conv2d(256, 256, 3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)

		# conv4
		self.conv4 = nn.Sequential(
			# max pooling
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
			# 512x512
			nn.Conv2d(256, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			# 512x512
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			# 512x512
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)

		# conv5
		self.conv5 = nn.Sequential(
			# max pooling
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
			# 512x512
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			# 512x512
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			# 512x512
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)
		return conv5

class HED_vgg16(nn.Module):
	def __init__(self):
		super(HED_vgg16, self).__init__()

		# conv1
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		# conv2
		self.conv2 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		# conv3
		self.conv3 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
			nn.Conv2d(128, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		# conv4
		self.conv4 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
			nn.Conv2d(256, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		# conv5
		self.conv5 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
		)

		self.dsn1 = nn.Conv2d(64, 1, 1)
		self.dsn2 = nn.Conv2d(128, 1, 1)
		self.dsn3 = nn.Conv2d(256, 1, 1)
		self.dsn4 = nn.Conv2d(512, 1, 1)
		self.dsn5 = nn.Conv2d(512, 1, 1)
		self.fuse = nn.Conv2d(5, 1, 1)

	def forward(self, x):
		h = x.size(2)
		w = x.size(3)

		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)

		## side output
		w1 = self.dsn1(conv1)
		w2 = F.upsample_bilinear(self.dsn2(conv2), size=(h,w))
		w3 = F.upsample_bilinear(self.dsn3(conv3), size=(h,w))
		w4 = F.upsample_bilinear(self.dsn4(conv4), size=(h,w))
		w5 = F.upsample_bilinear(self.dsn5(conv5), size=(h,w))

		A1 = w1 #F.relu(w1)
		A2 = w2 #F.relu(w2)
		A3 = w3 #F.relu(w3)
		A4 = w4 #F.relu(w4)
		A5 = w5 #F.relu(w5)

		# dsn fusion output
		fuse = self.fuse(torch.cat((A1, A2, A3, A4, A5), 1))
		
		Y1 = F.sigmoid(A1)
		Y2 = F.sigmoid(A2)
		Y3 = F.sigmoid(A3)
		Y4 = F.sigmoid(A4)
		Y5 = F.sigmoid(A5)
		Yfuse = F.sigmoid(fuse)

		return Y1, Y2, Y3, Y4, Y5, Yfuse

class HED_vgg16_bn(nn.Module):
	def __init__(self):
		super(HED_vgg16_bn, self).__init__()

		# conv1
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)

		# conv2
		self.conv2 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
			nn.Conv2d(64, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)

		# conv3
		self.conv3 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
			nn.Conv2d(128, 256, 3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)

		# conv4
		self.conv4 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
			nn.Conv2d(256, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)

		# conv5
		self.conv5 = nn.Sequential(
			nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)

		self.dsn1 = nn.Conv2d(64, 1, 1)
		self.dsn2 = nn.Conv2d(128, 1, 1)
		self.dsn3 = nn.Conv2d(256, 1, 1)
		self.dsn4 = nn.Conv2d(512, 1, 1)
		self.dsn5 = nn.Conv2d(512, 1, 1)
		self.fuse = nn.Conv2d(5, 1, 1)

	def forward(self, x):
		h = x.size(2)
		w = x.size(3)

		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)

		## side output
		d1 = self.dsn1(conv1)
		d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h,w))
		d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h,w))
		d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h,w))
		d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h,w))

		# dsn fusion output
		fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))
		
		d1 = F.sigmoid(d1)
		d2 = F.sigmoid(d2)
		d3 = F.sigmoid(d3)
		d4 = F.sigmoid(d4)
		d5 = F.sigmoid(d5)
		fuse = F.sigmoid(fuse)

		return d1, d2, d3, d4, d5, fuse

class HED_resnet(nn.Module):
	def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
		super(HED_resnet, self).__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		#self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		#self.fc = nn.Linear(512 * block.expansion, num_classes)

		self.dsn1 = nn.Conv2d(64, 1, 1)
		self.dsn2 = nn.Conv2d(128, 1, 1)
		self.dsn3 = nn.Conv2d(256, 1, 1)
		self.dsn4 = nn.Conv2d(512, 1, 1)
		#self.dsn5 = nn.Conv2d(512, 1, 1)
		self.fuse = nn.Conv2d(4, 1, 1)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		h = x.size(2)
		w = x.size(3)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		print("x.shape: ", x.shape)

		conv1 = self.layer1(x)
		print("conv1.shape:", conv1.shape)
		conv2 = self.layer2(conv1)
		conv3 = self.layer3(conv2)
		conv4 = self.layer4(conv3)

		# x = self.avgpool(x)
		# x = x.view(x.size(0), -1)
		# x = self.fc(x)

		## side output
		d1 = self.dsn1(conv1)
		d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h,w))
		d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h,w))
		d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h,w))

		# dsn fusion output
		fuse = self.fuse(torch.cat((d1, d2, d3, d4), 1))
		
		d1 = F.sigmoid(d1)
		d2 = F.sigmoid(d2)
		d3 = F.sigmoid(d3)
		d4 = F.sigmoid(d4)
		fuse = F.sigmoid(fuse)

		return d1, d2, d3, d4, fuse

def HED_resnet50(pretrained=False, **kwargs):
	model = HED_resnet(Bottleneck, [3,4,6,3], **kwargs)
	if pretrained == True:
		raise NotImplementedError
	return model

def HED_resnet101(pretrained=False, **kwargs):
	model = HED_resnet(Bottleneck, [3,4,23,3], **kwargs)
	if pretrained == True:
		raise NotImplementedError
	return model