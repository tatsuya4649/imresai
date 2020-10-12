"""
this file is to change Conv2d => PConv2d
-> https://qiita.com/saneatsu/items/da06e8632c7f5ba65279
"""
import torch
import torch.nn as nn
from utils.weight.weight_init import _setting_init,weight_init

class PartialConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode="zeros"):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.mask_conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,False,padding_mode)
        self.init_weight()
        for param in self.mask_conv.parameters():
            param.require_grad = False
    @_setting_init
    def init_weight(self):
        self.input_conv.apply(weight_init('kaiming')) # kaiming
        torch.nn.init.constant_(self.mask_conv.weight,1.0) # set mask_conv weight value = 1.0
    def forward(self,input,mask):
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1,-1,1,1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0.
        mask_sum = output_mask.masked_fill_(no_update_holes,1.0)
        output_pre = (output - output_bias)/mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes,0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes,0.0)
        return output,new_mask


if __name__ == "__main__":
    pconv = PConv2d(3,3,3)
    input = torch.rand(1,3,100,100)
    mask = torch.zeros_like(input)
    output,new_mask = pconv(input,mask)
