"""

test.py is to Image Restoration's test with PConvNet

"""

import sys,os
sys.path.append("net")
import torch
import torch.nn as nn
from net.pnet import PConvUNet
from utils.device import device
from utils.image import imread
from utils.mask import mask
from utils.norm import norm,unnorm
import argparse
from matplotlib import pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="test script of PConvNet")
parser.add_argument('-i','--image',help='test image "name",not "path"',default="moon.jpg")
parser.add_argument('-m','--model_path',help='PConvNet model parameters path',default="models/pnet.pth")
parser.add_argument('-s','--show',help='show result',default=True)
args = parser.parse_args()

_TEST_IMAGE_PATH = "images/{}".format(args.image)
_DEFAULT_MAX_SIZE = 1000
_MODEL_PATH = args.model_path

device = device()
net = PConvUNet()
ckpt_dict = torch.load(_MODEL_PATH,map_location=device)
net.load_state_dict(ckpt_dict['model'])
net = net.eval().to(device)
#making input tensor
test_tensor = imread(_TEST_IMAGE_PATH,_DEFAULT_MAX_SIZE).to(device)
test_tensor_ = norm(test_tensor)
mask = mask(test_tensor).to(device)
#prediction
with torch.no_grad():
	net_output,_ = net(test_tensor_,mask)

net_output = unnorm(net_output)
output = mask * test_tensor + (1 - mask) * net_output

if args.show:
    fig,axes = plt.subplots(figsize=(10,3),ncols=3)
    im1 = axes[0].imshow(test_tensor[0].cpu().detach().numpy().transpose(1,2,0))
    fig.colorbar(im1,ax=axes[0])
    im2 = axes[1].imshow(mask[0].cpu().detach().numpy().transpose(1,2,0))
    fig.colorbar(im2,ax=axes[1])
    im3 = axes[2].imshow(net_output[0].cpu().detach().numpy().transpose(1,2,0))
    fig.colorbar(im3,ax=axes[2])
    plt.show()
