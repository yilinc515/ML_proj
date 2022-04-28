from re import L
import torch
import numpy as np
import torch.nn.functional as F
from main_SupRes import N_IMAGES
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



# PSNR
# def PSNR(img1, img2):
    
#     return psnr(img1, img2)

def avg_PSNR(train_est : np.ndarray, train_true : np.ndarray):
    sum = 0
    for i in N_IMAGES:
        sum = sum + psnr(train_true[i], train_est[i])
    return sum/N_IMAGES

# SSIM
# def SSIM(img1, img2):
#     return ssim(img1, img2, channel_axis = #TODO)

def avg_SSIM(train_est : np.ndarray, train_true : np.ndarray):
    sum = 0
    for i in N_IMAGES:
        sum = sum + ssim(train_true[i], train_est[i], channel_axis = 0)
    return sum/N_IMAGES
 

def approx_train_psnr_ssim(model, train_comp : np.ndarray, train_true : np.ndarray) -> np.float64:
    idxs = np.random.choice(train_comp.shape[0], 4000, replace=False)
    x = torch.from_numpy(train_comp[idxs].astype(np.float32))
    y = torch.from_numpy(train_true[idxs].astype(np.float32))
    restored = model(x)
    train_psnr = avg_PSNR(restored, y)
    train_ssim = avg_SSIM(restored, y)
    return train_psnr, train_ssim


def dev_loss_psnr_ssim(model, dev_comp : np.ndarray, dev_true : np.ndarray) -> np.float64:
    x = torch.from_numpy(dev_comp.astype(np.float32))
    y = torch.from_numpy(dev_true.astype(np.float32))
    restored = model(x)
    #loss = F.cross_entropy(restored, y)
    loss = F.l1_loss(restored, y)
    dev_psnr = avg_PSNR(restored, y)
    dev_ssim = avg_SSIM(restored, y)
    return loss.item(), dev_psnr, dev_ssim

