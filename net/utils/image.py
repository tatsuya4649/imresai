

import torch
import cv2

def imread(path,max_size):
    im_cv2 = cv2.imread(path)
    im_np = cv2.cvtColor(im_cv2,cv2.COLOR_BGR2RGB)
    height = im_np.shape[0]
    width = im_np.shape[1]
    if height < width:
        im_np = cv2.resize(im_np,(int(max_size),int(max_size*(height/width))))
    else:
        im_np = cv2.resize(im_np,(int(max_size*(width/height)),int(max_size)))
    print("resize image {}x{} => {}x{} ".format(height,width,im_np.shape[0],im_np.shape[1]))
    im_tensor = torch.from_numpy(im_np)
    im_tensor = im_tensor.permute(2,0,1).unsqueeze(0).float()
    im_tensor /= 255.
    return im_tensor
