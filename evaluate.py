import os, sys
import torch

from model import *

def evaluate(opt):
	evalPath = opt.data_root+'train_pair.lst'

	# create data loaders from dataset
	transform = transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize(opt.mean,opt.std)
							])
	targetTransform = transforms.Compose([
									transforms.ToTensor()
							])

	evalDataset = TrainDataset(evalPath, opt.data_root, transform, targetTransform)
	evalDataloader = DataLoader(evalDataset, shuffle=False)

		# initialize the network
	if opt.arch == 'vgg16':
		net = HED_vgg16()
	elif opt.arch == 'vgg16_bn':
		net = HED_vgg16_bn()
	else:
		raise NotImplementedError

	