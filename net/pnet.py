
import torch
import torch.nn as nn
from pblock import PConvBlock
from pconv import PConv2d

class PConvNet(nn.Module):
    def __init__(self,layer_size=7,in_channels=3,upsample_mode="nearest"):
        super().__init__()
        self._upsample_mode = upsample_mode
        self._layer_size = layer_size
        self.enc_1 = PConvBlock(in_channels,64,False,step_name="down_7")
        self.enc_2 = PConvBlock(64,128,step_name="down_5")
        self.enc_3 = PConvBlock(128,256,step_name="down_5")
        self.enc_4 = PConvBlock(256,512,step_name="down_3")
        for i in range(4,self.layer_size):
            name = 'enc_{}'.format(i+1)
            setattr(self,name,PConvBlock(512,512,step_name="down_3"))

        for i in range(4,self.layer_size):
            name = 'dec_{}'.format(i+1)
            setattr(self,name,PConvBlock(512+512,512,activation_name='leaky'))

        self.dec_4 = PConvBlock(512+256,256,activation_name='leaky')
        self.dec_3 = PConvBlock(256+128,128,activation_name='leaky')
        self.dec_2 = PConvBlock(128+64,64,activation_name='leaky')
        self.dec_1 = PConvBlock(64+in_channels,in_channels,batch_norm=False,activation_name=None,conv_bias=True)
        self.upsample = nn.Upsample(scale_factor=2,mode=self._upsample_mode)

    @property
    def layer_size(self):
        return self._layer_size

    def forward(self,input,input_mask):
        enc_dict = dict()
        enc_mask_dict = dict()

        enc_dict["enc_0"],enc_mask_dict["enc_0"] = input,input_mask
        enc_key_pre = "enc_0"
        for i in range(1,self.layer_size+1):
            enc_key = "enc_{}".format(i)
            enc_dict[enc_key],enc_mask_dict[enc_key] = getattr(self,enc_key)(enc_dict[enc_key_pre],enc_mask_dict[enc_key_pre])
            enc_key_pre = enc_key
        
        output,mask = enc_dict[enc_key],enc_mask_dict[enc_key]
        for i in range(self.layer_size,0,-1):
            enc_key = "enc_{}".format(i-1)
            dec_key = "dec_{}".format(i)
            output = self.upsample(output)
            mask = self.upsample(mask)
            output = torch.cat([output,enc_dict[enc_key]],dim=1)
            mask = torch.cat([mask,enc_mask_dict[enc_key]],dim=1)
            output,mask = getattr(self,dec_key)(output,mask)
        return output,mask

if __name__ == "__main__":
    net = PConvNet()
    rand_input = torch.rand(1,3,512,512)
    rand_mask = torch.zeros_like(rand_input)
    output,mask = net(rand_input,rand_mask)
    print(output.shape)
    print(mask.shape)
