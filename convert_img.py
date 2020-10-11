"""
Converting Between Different Image File Formats

.png - 3*H*W (channels, height, width), in RGB format
.raw - 1*H*W with bayer filter pattern (GBRG)
.mat - 4*H*W, in RGGB format

"""

import numpy as np
from scipy.io import loadmat, savemat
from PIL import Image
import cv2
import pickle

def cvtpng2raw(pwd):
    # Takes in .png image file and saves it as a .raw2 file format
    img = Image.open(pwd)
    data = np.asarray(img)
    raw = rgb2raw(data, bayer_pattern='gbrg')				# Provided subroutine that converts 3 channel RGB image to 1 channel bayer (GBRG) image
    raw = raw.astype('uint16')*4
    raw.astype('int16').tofile(pwd.replace('.png', '.raw2'))

def cvtpng2mat(pwd):
    # Takes in .png image file and saves it as a .mat file format
    img = Image.open(pwd)
    data = np.asarray(img)
    raw = rgb2raw(data.copy(), bayer_pattern='gbrg')			# Provided subroutine that converts 3 channel RGB image to 1 channel bayer (GBRG) image
    raw1 = raw[1:,:]
    H,W = raw1.shape
    raw1 = raw1.reshape(H//2,2,W//2,2)
    raw1 = raw1.transpose(0,2,1,3)
    raw1 = raw1.reshape(H//2,W//2,4)
    savemat(pwd.replace('.png', '.mat'), mdict={'ps4k': raw1})
    
def cvtraw2png(pwd, h, w):
    # Takes in .raw2 image file and saves it as a .png file format
    raw = np.fromfile(pwd, dtype=np.uint16).reshape((h,w))
    opt = cv2.COLOR_BAYER_GR2BGR
    img = cv2.cvtColor(raw, opt)
    max_intensity = np.ndarray.max(raw)
    img = img / max_intensity * 255. + 0.5
    img = img.astype(np.uint8)
    cv2.imwrite(pwd.replace('.raw2', '.png'), img)
    
def cvtraw2mat(pwd, h, w):
    # Takes in .raw2 image file and saves it as a .mat file format
    raw = np.fromfile(pwd, dtype=np.uint16).reshape((h,w))
    raw1 = raw[1:,:]
    raw1 = np.vstack((raw1, np.zeros(w).astype(np.uint16)))
    H,W = raw1.shape
    raw1 = raw1.reshape(H//2,2,W//2,2)
    raw1 = raw1.transpose(0,2,1,3)
    raw1 = raw1.reshape(H//2,W//2,4)
    savemat(pwd.replace('.raw2', '.mat'), mdict={'ps4k': raw1})

def cvtmat2png(pwd):
    # Takes in .mat image file and saves it as a .png file format
    mat = loadmat(pwd)
    arr = mat['ps4k']
    H,W,C = arr.shape
    max_intensity = np.ndarray.max(arr)
    bayerImg = arr.reshape([H,W,2,2])
    bayerImg = bayerImg.transpose([0,2,1,3])
    bayerImg = bayerImg.reshape([H*2,W*2]).astype(np.uint16)
    img = cv2.cvtColor(bayerImg, cv2.COLOR_BayerGR2BGR)
    img = img / max_intensity * 255. + 0.5
    max_intensity = np.ndarray.max(arr)
    img = img.astype(np.uint8)
    cv2.imwrite(pwd.replace('.mat', '.png'), img)

def cvtmat2raw(pwd):
    # Takes in .mat image file and saves it as a .raw2 file format
    mat = loadmat(pwd)
    arr = mat['ps4k']
    H,W,C = arr.shape  
    raw = rggb_prepro(img, scale)					# Provided subroutine that converts 4 channel bayer image into 1 channel bayer image
    raw1 = raw.numpy().astype('uint16')
    raw.astype('int16').tofile(pwd.replace('.mat', '.raw2'))

def cvtpng2pt(pwd):
    # Takes in .png image file and saves it as a .pt file format
    img = Image.open(pwd)
    data = np.asarray(img)
    pickle.dump(data, pwd.replace('.png', '.pt'))

def cvtpt2png(pwd):
    # Takes in .pt image file and saves it as a .png file format
    f = open(pwd, 'rb')
    img = pickle.load(f)
    cv2.imwrite(pwd.replace('.pt', '.png'), img)
    
    
