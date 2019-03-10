# main program to train hed model
# Author: Nishanth
# Date: 2017/10/19

# import torch libraries
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# import utility functions
from model import *
from trainer import Trainer
from dataproc import TrainDataset
from options import TrainOptions

# fix random seed
rng = np.random.RandomState(37148)

# GPU ID
# gpuID = 0

# batch size
# nBatch = 1

# load the images dataset
# dataRoot = 'data/HED-BSDS/'
# modelPath = 'model/vgg16_pth-IN.pth'

def train(opt):
	valPath   = opt.data_root+'val_pair.lst'
	trainPath = opt.data_root+'train_pair.lst'

	# create data loaders from dataset
	transform = transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize(opt.mean,opt.std)
							])
	targetTransform = transforms.Compose([
									transforms.ToTensor()
							])

	valDataset = TrainDataset(valPath, opt.data_root, transform, targetTransform)
	trainDataset = TrainDataset(trainPath, opt.data_root, transform, targetTransform)

	valDataloader = DataLoader(valDataset, shuffle=False)
	trainDataloader = DataLoader(trainDataset, shuffle=True)

	# initialize the network
	if opt.arch == 'vgg16':
		net = HED_vgg16()
	elif opt.arch == 'vgg16bn':
		net = HED_vgg16_bn()
	else:
		raise NotImplementedError

	net.apply(weights_init)

	pretrained_dict = torch.load(opt.bb_weight)
	# if opt.arch == 'vgg16':
	# 	# pretrained_dict = convert_pth_vgg16(pretrained_dict)
	# 	pretrained_dict = state_dict_pth2custom(pretrained_dict, custom_vgg16)
	# elif opt.arch == 'vgg16_bn':
	# 	# pretrained_dict = convert_pth_vgg16_bn(pretrained_dict)
	# 	pretrained_dict = state_dict_pth2custom(pretrained_dict, custom_vgg16_bn) 
	# else:
	# 	raise NotImplementedError

	net.load_state_dict(pretrained_dict, strict=False)

	# model_dict = net.state_dict()
	# model_dict.update(pretrained_dict)
	# net.load_state_dict(model_dict)

	net = net.to(opt.device)
	if opt.cuda and torch.cuda.device_count() > 1:
		net = nn.DataParallel(net)

	# define the optimizer
	# lr = 1e-4
	# lrDecay = 1e-1
	lrDecayEpoch = {3,5,8,10,12}

	fuse_params = list(map(id, net.fuse.parameters()))
	conv5_params = list(map(id, net.conv5.parameters()))
	base_params = filter(lambda p: id(p) not in conv5_params+fuse_params,
											net.parameters())

	optimizer = torch.optim.SGD([
							{'params': base_params},
							{'params': net.conv5.parameters(), 'lr': opt.lr * 100},
							{'params': net.fuse.parameters(), 'lr': opt.lr * 0.001}
							], lr=opt.lr, momentum=opt.momentum)

	# initialize trainer class
	trainer = Trainer(net, optimizer, trainDataloader, valDataloader, 
										# nBatch=opt.batch_size, maxEpochs=15, cuda=True, gpuID=gpuID,
										opt, lrDecayEpochs=lrDecayEpoch)

	# train the network
	trainer.train()

if __name__ == "__main__":
 opt = TrainOptions().parse()
 train(opt)