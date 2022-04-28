from skimage.transform import resize
import os
import numpy as np
from utils.accuracies import (avg_PSNR, avg_SSIM)
import matplotlib.pyplot as plt


DATA_DIR = "datasets/"
print("Loading from", DATA_DIR)
TRAIN_COMP_IMAGES = np.load(os.path.join(DATA_DIR, "train_comp_images.npy"))
TRAIN_TRUE_IMAGES = np.load(os.path.join(DATA_DIR, "train_true_images.npy"))
DEV_COMP_IMAGES = np.load(os.path.join(DATA_DIR, "dev_comp_images.npy"))
DEV_TRUE_IMAGES = np.load(os.path.join(DATA_DIR, "dev_true_images.npy"))

################
# Train images
################
# interpolation
print(TRAIN_COMP_IMAGES.shape)
print(TRAIN_TRUE_IMAGES.shape)
plt.imshow(np.moveaxis(TRAIN_COMP_IMAGES[0], 0, -1))
print("Computing train")
bilinear = []
bicubic = []
for i in TRAIN_COMP_IMAGES:
    l = resize(i, (3, 224, 224), order = 1, preserve_range=True)  #bilinear interpolation
    l = l.astype(np.int64)
    bilinear.append(l)
    c = resize(i, (3, 224, 224), order = 3, preserve_range=True)  #bicubic interpolation
    c = c.astype(np.int64)
    bicubic.append(c)
bilinear = np.array(bilinear)
print(bilinear.shape)
plt.imshow(np.moveaxis(bilinear[0], 0, -1))
bicubic = np.array(bicubic)

# compute PSNR and SSIM
# bilinear
bilinear_psnr = avg_PSNR(bilinear, TRAIN_TRUE_IMAGES)
bilinear_ssim = avg_SSIM(bilinear, TRAIN_TRUE_IMAGES)
print("Bi-linear interpolation train PSNR: ", bilinear_psnr )
print("Bi-linear interpolation train SSIM: ", bilinear_ssim )
# bicubic
bicubic_psnr = avg_PSNR(bicubic , TRAIN_TRUE_IMAGES)
bicubic_ssim = avg_SSIM(bicubic , TRAIN_TRUE_IMAGES)
print("Bi-bicubic interpolation train PSNR: ", bicubic_psnr )
print("Bi-bicubic interpolation train SSIM: ", bicubic_ssim )





################
# Dev images
################
# interpolation
bilinear = []
bicubic = []
for i in DEV_COMP_IMAGES:
    l = resize(i, (3, 224, 224), order = 1, preserve_range=True)  #bilinear interpolation
    l = l.astype(np.int64)
    bilinear.append(l)
    c = resize(i, (3, 224, 224), order = 3, preserve_range=True)  #bicubic interpolation
    c = c.astype(np.int64)
    bicubic.append(c)
bilinear = np.array(bilinear)
bicubic = np.array(bicubic)

# compute PSNR and SSIM
# bilinear
bilinear_psnr = avg_PSNR(bilinear, DEV_TRUE_IMAGES)
bilinear_ssim = avg_SSIM(bilinear, DEV_TRUE_IMAGES)
print("Bi-linear interpolation dev PSNR: ", bilinear_psnr )
print("Bi-linear interpolation dev SSIM: ", bilinear_ssim )
# bicubic
bicubic_psnr = avg_PSNR(bicubic , DEV_TRUE_IMAGES)
bicubic_ssim = avg_SSIM(bicubic , DEV_TRUE_IMAGES)
print("Bi-bicubic interpolation dev PSNR: ", bicubic_psnr )
print("Bi-bicubic interpolation dev SSIM: ", bicubic_ssim )