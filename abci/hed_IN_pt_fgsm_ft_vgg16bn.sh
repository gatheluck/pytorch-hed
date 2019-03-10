#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N hed_IN_pt_fgsm_ft_vgg16bn
#$ -o /home/aaa10329ah/user/waseda/abci_log/hed_IN_pt_fgsm_ft_vgg16bn.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate faster-rcnn.pytorch
cd /home/aaa10329ah/user/waseda/faster-rcnn
# script

python train.py -a vgg16 \
								-j 40 \
								-b 10 \
								--lr 1e-6 \
								--wd 0.0002 \
								--num_epochs 10000 \
								--checkpoint 5 \
								--bb_weight ./data/models/IN_pt_fgsm_ft_vgg16bn.pth \
								-l ./logs/hed_IN_pt_fgsm_ft_vgg16bn \
								-r result.json \
								--cuda \
								--mGPUs