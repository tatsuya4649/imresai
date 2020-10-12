

import torch
import torch.nn as nn
from pconv import PConv2d

class PConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,batch_norm=True,step_name='',conv_bias=False,activation_name='relu'):
        super().__init__()
        if step_name == 'down_5':
            self.conv = PConv2d(in_channels,out_channels,5,2,2,bias=conv_bias)
        elif step_name == 'down_7':
            self.conv = PConv2d(in_channels,out_channels,7,2,3,bias=conv_bias)
        elif step_name == 'down_3':
            self.conv = PConv2d(in_channels,out_channels,3,2,1,bias=conv_bias)
        else:
            self.conv = PConv2d(in_channels,out_channels,3,1,1,bias=conv_bias)

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        if activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self,input,mask):
        output,mask_output = self.conv(input,mask)
        if hasattr(self,'batch_norm'):
            output = self.batch_norm(output)
        if hasattr(self,'activation'):
            output = self.activation(output)
        return output,mask_output

if __name__ == "__main__":
    pblock = PConvBlock(3,3,True,"down_5",False)
    rand_input = torch.rand(1,3,100,100)
    rand_mask = torch.zeros_like(rand_input)
    output,mask_output = pblock(rand_input,rand_mask)
    print(mask_output)
