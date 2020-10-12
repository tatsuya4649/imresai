import torch
import torchvision
_MEAN = torch.Tensor([0.485,0.456,0.406])
_STD = torch.Tensor([0.229,0.224,0.225])

def norm(tensor):
    results = list()
    for i in range(tensor.shape[0]):
        transform = torchvision.transforms.Normalize(_MEAN,_STD)(tensor[i])
        results.append(transform)
    return torch.cat(results).reshape(len(results),*results[0].shape)

def unnorm(tensor):
    tensor = tensor.transpose(1,3)
    tensor = tensor * _STD + _MEAN
    tensor = tensor.transpose(1,3)
    return tensor
    
