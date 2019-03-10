# trainer.py: Utility function to train HED model
# Author: Nishanth Koganti
# Date: 2017/10/20

# Source: https://github.com/xlliu7/hed.pytorch/blob/master/trainer.py

# Issues:
# 

# import libraries
import math
import os, time
import json
import numpy as np
from PIL import Image
import os.path as osp
from tqdm import tqdm

# import torch modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# utility class for training HED model
class Trainer(object):
	# init function for class
	def __init__(self, generator, optimizerG, trainDataloader, valDataloader,
							#nBatch=10, out='train', maxEpochs=1, cuda=True, gpuID=0,
							opt, lrDecayEpochs={}):

		# set the GPU flag
		# self.cuda = cuda
		# self.gpuID = gpuID
		self.log_dir = opt.log_dir
		self.device = opt.device
		
		# define an optimizer
		self.optimG = optimizerG
		
		# set the network
		self.generator = generator
		
		# set the data loaders
		self.valDataloader = valDataloader
		self.trainDataloader = trainDataloader
		
		# set output directory
		# self.out = out
		# if not osp.exists(self.out):
		#     os.makedirs(self.out)
						
		# set training parameters
		self.epoch = 0
		self.nBatch = opt.batch_size
		self.nepochs = opt.num_epochs
		self.lrDecayEpochs = lrDecayEpochs
		
		self.gamma = opt.gamma
		self.valInterval = opt.print_freq
		self.dispInterval = opt.print_freq
		self.timeformat = '%Y-%m-%d %H:%M:%S'

		self.opt = opt

	def train(self):
		best_loss = np.inf
		curr_loss = np.inf
		best_epoch = 0

		# function to train network
		for epoch in range(self.epoch, self.nepochs):
			
			self.generator.train() # set function to training mode
			self.optimG.zero_grad() # initialize gradients
			
			# adjust hed learning rate
			if epoch in self.lrDecayEpochs:
				self.adjustLR()

			# train the network
			losses = []
			lossAcc = 0.0
			tbar = tqdm(self.trainDataloader)
			for i, sample in enumerate(tbar, 0):
				# get the training batch
				data, target = sample
				data, target = data.to(self.device), target.cuda(self.device)
				data, target = Variable(data), Variable(target)
				
				# generator forward
				if self.opt.arch == 'vgg16' or self.opt.arch == 'vgg16bn':
					Y = target
					Y1, Y2, Y3, Y4, Y5, Yfuse = self.generator(data) 
					
					# compute loss for batch
					loss_1 = self.bce2d(Y1, Y)
					loss_2 = self.bce2d(Y2, Y)
					loss_3 = self.bce2d(Y3, Y)
					loss_4 = self.bce2d(Y4, Y)
					loss_5 = self.bce2d(Y5, Y)
					loss_fuse = self.bce2d(Yfuse, Y)
					
					loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_fuse # this part is different from original paper
					if np.isnan(float(loss.item())): raise ValueError('loss is nan while training')
				else:
					Y = target
					Y1, Y2, Y3, Y4, Yfuse = self.generator(data) 
					
					# compute loss for batch
					loss_1 = self.bce2d(Y1, Y)
					loss_2 = self.bce2d(Y2, Y)
					loss_3 = self.bce2d(Y3, Y)
					loss_4 = self.bce2d(Y4, Y)
					loss_fuse = self.bce2d(Yfuse, Y)
					
					loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_fuse # this part is different from original paper
					if np.isnan(float(loss.item())): raise ValueError('loss is nan while training')

				losses.append(loss)
				lossAcc += loss.item()
				curr_loss += loss.item()
						
				# perform backpropogation and update network
				if i%self.nBatch == 0:
					bLoss = sum(losses)
			
					bLoss.backward()
					self.optimG.step()
					self.optimG.zero_grad()

					losses = []
						
				# visualize the loss
				if (i+1) % self.dispInterval == 0:
					timestr = time.strftime(self.timeformat, time.localtime())
					print("%s epoch: %d iter:%d loss:%.6f"%(timestr, epoch+1, i+1, lossAcc/self.dispInterval))
					lossAcc = 0.0
						
				# perform validation every 500 iters
				if (i+1) % self.valInterval == 0:
					self.val(epoch+1)
					if curr_loss < best_loss:
						best_loss = curr_loss
						best_epoch = epoch
						torch.save(self.generator.module.state_dict() if isinstance(self.generator, nn.DataParallel) else self.generator.state_dict(), self.log_dir+"weight_best.pth")
						# save result
						save_result({
							'loss': best_loss,
						}, self.opt.log_dir, self.opt.result+'_best.json')
						curr_loss = 0.0
								
		# save model after every epoch
		torch.save(self.generator.module.state_dict() if isinstance(self.generator, nn.DataParallel) else self.generator.state_dict(), self.log_dir+"weight_final.pth")
		# save result
		save_result({
			'loss': curr_loss,
		}, self.opt.log_dir, self.opt.result+'_final.json')

	def val(self, epoch):
		print('Evaluation:') # eval model on validation set
		self.generator.eval()
		
		# save the results
		if not os.path.exists(self.log_dir + '/images'): os.mkdir(self.log_dir + '/images')
		dirName = self.log_dir+'/images'
		
		# perform test inference
		for i, sample in enumerate(self.valDataloader, 0):            
			# get the test sample
			data, target = sample
			data, target = data.to(self.device), target.to(self.device)
			data, target = Variable(data), Variable(target)
			
			if self.opt.arch == 'vgg16' or self.opt.arch == 'vgg16bn':
				# perform forward computation
				d1, d2, d3, d4, d5, d6 = self.generator.forward(data)
				
				# transform to grayscale images
				d1 = self.grayTrans(self.crop(d1))
				d2 = self.grayTrans(self.crop(d2))
				d3 = self.grayTrans(self.crop(d3))
				d4 = self.grayTrans(self.crop(d4))
				d5 = self.grayTrans(self.crop(d5))
				d6 = self.grayTrans(self.crop(d6))
				tar = self.grayTrans(self.crop(target))
				
				d1.save('{}/sample{:02d}_1.png'.format(dirName, i))
				d2.save('{}/sample{:02d}_2.png'.format(dirName, i))
				d3.save('{}/sample{:02d}_3.png'.format(dirName, i))
				d4.save('{}/sample{:02d}_4.png'.format(dirName, i))
				d5.save('{}/sample{:02d}_5.png'.format(dirName, i))
				d6.save('{}/sample{:02d}_6.png'.format(dirName, i))
				tar.save('{}/sample{:02d}_T.png'.format(dirName, i))
			else:
				# perform forward computation
				d1, d2, d3, d4, d5 = self.generator.forward(data)
				
				# transform to grayscale images
				d1 = self.grayTrans(self.crop(d1))
				d2 = self.grayTrans(self.crop(d2))
				d3 = self.grayTrans(self.crop(d3))
				d4 = self.grayTrans(self.crop(d4))
				d5 = self.grayTrans(self.crop(d5))
				tar = self.grayTrans(self.crop(target))
				
				d1.save('{}/sample{:02d}_1.png'.format(dirName, i))
				d2.save('{}/sample{:02d}_2.png'.format(dirName, i))
				d3.save('{}/sample{:02d}_3.png'.format(dirName, i))
				d4.save('{}/sample{:02d}_4.png'.format(dirName, i))
				d5.save('{}/sample{:02d}_5.png'.format(dirName, i))
				tar.save('{}/sample{:02d}_T.png'.format(dirName, i))

		print('evaluate done')
		self.generator.train()
    
	# function to crop the padding pixels
	def crop(self, d):
		d_h, d_w = d.size()[2:4]
		g_h, g_w = d_h-64, d_w-64
		d1 = d[:, :, int(math.floor((d_h - g_h)/2.0)):int(math.floor((d_h - g_h)/2.0)) + g_h, int(math.floor((d_w - g_w)/2.0)):int(math.floor((d_w - g_w)/2.0)) + g_w]
		return d1
	
	def _assertNoGrad(self, variable):
		assert not variable.requires_grad, \
		"nn criterions don't compute the gradient w.r.t. targets - please " \
		"mark these variables as volatile or not requiring gradients"

	# binary cross entropy loss in 2D
	def bce2d(self, input, target):
		n, c, h, w = input.size()

		# assert(max(target) == 1)
		log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
		target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
		target_trans = target_t.clone()
		pos_index = (target_t >0)
		neg_index = (target_t ==0)
		target_trans[pos_index] = 1
		target_trans[neg_index] = 0
		pos_index = pos_index.data.cpu().numpy().astype(bool)
		neg_index = neg_index.data.cpu().numpy().astype(bool)
		weight = torch.Tensor(log_p.size()).fill_(0)
		weight = weight.numpy()
		pos_num = pos_index.sum()
		neg_num = neg_index.sum()
		sum_num = pos_num + neg_num
		weight[pos_index] = neg_num*1.0 / sum_num
		weight[neg_index] = pos_num*1.0 / sum_num

		weight = torch.from_numpy(weight)
		weight = weight.cuda()
		loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
		return loss

	def grayTrans(self, img):
		img = img.data.cpu().numpy()[0][0]*255.0
		img = (img).astype(np.uint8)
		img = Image.fromarray(img, 'L')
		return img

	# utility functions to set the learning rate
	def adjustLR(self):
		for param_group in self.optimG.param_groups:
			param_group['lr'] *= self.gamma 

def save_result(result, log_dir, filename):
	path = os.path.join(log_dir, filename)
	dir = os.path.dirname(path)
	os.makedirs(dir, exist_ok=True)

	with open(path, 'w') as f:
		f.write(json.dumps(result, indent=4))