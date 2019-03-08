import os
import sys
import argparse

import torch
from torchvision import models

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

model_names = ['vgg16', 'vgg16_bn']
dataset_names = ['bsds']

class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		# model
		parser.add_argument('-a', '--arch', type=str, required=True, choices=model_names, help='model architecture: ' + ' | ' .join(model_names), metavar='ARCH')
		# dataset
		parser.add_argument('-d', '--dataset', type=str, default='bsds', choices=dataset_names, help='dataset: ' + ' | '.join(dataset_names), metavar='DATASET')
		parser.add_argument('-j', '--num_workers', type=int, default=4, help='number of workers for data loading')
		parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
		# GPU
		parser.add_argument('--cuda', action='store_true', default=False, help='enable GPU')
		# log
		parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
		parser.add_argument('-l', '--log_dir', type=str, required=True, help='log directory')
		parser.add_argument('-r', '--result', type=str, required=True, help='result json path')

		self.initialized = True
		return parser

	def gather_options(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '---------------------------- Options --------------------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: {}]'.format(str(default))
			message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
		message += '---------------------------- End ------------------------------'
		print(message)

		os.makedirs(opt.log_dir, exist_ok=True)
		with open(os.path.join(opt.log_dir, 'options.txt'), 'wt') as f:
			command = ''
			for k, v in sorted(vars(opt).items()):
				command += '--{} {} '.format(k, str(v))
			command += '\n'
			f.write(command)
			f.write(message)
			f.write('\n')
	
	def parse(self):
		opt = self.gather_options()
		self.print_options(opt)

		# GPU
		if opt.cuda and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			opt.device = 'cuda'
		else:
			opt.cuda = False
			opt.device = 'cpu'

		self.opt = opt
		return self.opt
			

class TrainOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)

		# model and backbone
		parser.add_argument('--bb_weight', type=str, required=True, help='path to pretrained backbone weight')
		parser.add_argument('--checkpoint', type=int, default=5, help='checkpoint epoch')
		parser.add_argument('--resume', type=int, default=-1, help='resume epoch for model loading')
		# hyperparameter and optimaizer
		parser.add_argument('--optimizer', type=str, default='sgd', help='training optimizer')
		parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
		parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
		parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
		parser.add_argument('--wd', type=float, default=1e-1, help='weight decay')
		parser.add_argument('--step_size', type=int, default=5, help='step size for scheduler')
		parser.add_argument('--gamma', type=float, default=0.1, help='gamma for scheduler')
		

		return parser