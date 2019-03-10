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

# def convert_pth_vgg16(vgg_pth):
# 	net = vgg16()
# 	vgg_own_items = list(net.state_dict().items())
# 	vgg_pth_items = list(vgg_pth.items())
# 	pretrain_model = {}
# 	j = 0
# 	for k, v in net.state_dict().items():
# 		print("k_pth:", k)
# 		v = vgg_pth_items[j][1]
# 		#print("v:", v)
# 		k = vgg_own_items[j][0]
# 		print("k_own:", k)
# 		pretrain_model[k] = v
# 		j += 1
# 	return pretrain_model

# def convert_pth_vgg16_bn(vgg_pth):
# 	net = custom_vgg16_bn()
# 	vgg_own_items = list(net.state_dict().items())
# 	vgg_pth_items = list(vgg_pth.items())
# 	pretrain_model = {}
# 	j = 0
# 	for k, v in net.state_dict().items():
# 		print("k_pth:", k)
# 		v = vgg_pth_items[j][1]
# 		# print("v:", v)
# 		k = vgg_own_items[j][0]
# 		print("k_own:", k)
# 		pretrain_model[k] = v
# 		j += 1
# 	return pretrain_model

# def state_dict_pth2custom(pth_state_dict, cst_model_class):
# 	cst_model = cst_model_class()
# 	cst_items = list(cst_model.state_dict().items())
# 	pth_items = list(pth_state_dict.items())
# 	print("cst_items[:][0]:", cst_items[:][0])
# 	print("pth_items[:][0]:", pth_items[:][0])
# 	print("len(cst_items):", len(cst_items))
# 	print("len(pth_items):", len(pth_items))

# 	for i in range(len(cst_items)):
# 		key = cst_items[i][0]
# 		if 'num_batches_tracked' in key:
# 			del cst_items[i][0]

# 	for i in range(len(cst_items)):
# 		print("cst_items[{}][0]".format(i), cst_items[i][0])

# 	for i in range(len(pth_items)):
# 		print("pth_items[{}][0]".format(i), pth_items[i][0])

# 	state_dict = {}
# 	idx_cst = 0
# 	idx_pth = 0
# 	for k, v in cst_model.state_dict().items():
# 		if 'num_batches_tracked' in k:
# 			idx_cst
# 		k = cst_items[j][0]
# 		v = pth_items[j][1]
# 		state_dict[k] = v
# 		j += 1
# 	return state_dict



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
