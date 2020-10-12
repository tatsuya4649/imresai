
import torch
import torch.nn as nn
import math


def weight_init(init_type='guassian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if "Conv2d" in classname and hasattr(m,'weight'):
            if init_type == 'guassian':
                nn.init.normal_(m.weight,0.0,0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight,gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight,gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                raise ValueError("{} is Unsupport init_type".format(m))
            if hasattr(m,'bias') and m.bias is not None:
                nn.init.constant_(m.bias,0.0)
    return init_fun

def _setting_init(func):
    def notification(*args,**kwargs):
        func(*args,**kwargs)
        print("{} => setting weight init".format(func.__name__))
    return notification
