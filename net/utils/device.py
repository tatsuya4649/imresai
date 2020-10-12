import torch

def device_print(func):
    def _print(*args,**kwargs):
        print('========================')
        device = func(*args,**kwargs)
        print(' device => {}'.format(device))
        print('========================')
        return device
    return _print

@device_print
def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

