import torch
from matplotlib import pyplot as plt

def mask(input,ratio=0.05,show_mask=False):
    input_ones = torch.ones_like(input)
    height = input_ones.shape[2]
    width = input_ones.shape[3]
    mask_height = int(height*ratio)
    mask_width = int(width*ratio)
    input_ones[:,:,int(height/2 - mask_height/2):int(height/2 + mask_height/2),int(width/2 - mask_width/2):int(width/2 + mask_width/2)] = 0.
    if show_mask:
        input_ = input[0].cpu().detach().numpy().transpose(1,2,0)
        mask_ = input_ones[0].cpu().detach().numpy().transpose(1,2,0)
        plt.imshow(input_*mask_)
        plt.show()
    return input_ones.float()

if __name__ == "__main__":
    mask(torch.rand(1,3,512,512),show_mask=True)
