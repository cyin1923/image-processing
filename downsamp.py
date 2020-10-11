import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat
import cv2

"""
Input: pwd - directory path to input .mat file
Performs average pooling and saves as .png
"""
def avgpool(pwd):
    mat = loadmat(pwd)
    arr = mat['ps4k']
    np_arr = arr.astype(np.float32)
    np_arr = np_arr.transpose(2,0,1)

    pool_arr = F.avg_pool2d(torch.from_numpy(np_arr), 2, 2)

    C,H,W = pool_arr.shape
    max_intensity = np.ndarray.max(arr)

    bayerImg = pool_arr.numpy().transpose(1,2,0).reshape([H,W,2,2])
    bayerImg = bayerImg.transpose(0,2,1,3)
    bayerImg = bayerImg.reshape([H*2,W*2]).astype(np.uint16)
    img = cv2.cvtColor(bayerImg, cv2.COLOR_BAYER_BG2BGR)
    img = img / max_intensity * 255. + 0.5
    img = img.astype(np.uint8)
    cv2.imwrite(pwd.replace('.mat', '.png', img)

