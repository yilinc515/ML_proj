from re import L
import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def accuracy(y : np.ndarray, y_hat : np.ndarray) -> np.float64:
    """Calculate the simple accuracy given two numpy vectors, each with int values
    corresponding to each class.

    Args:
        y (np.ndarray): actual value
        y_hat (np.ndarray): predicted value

    Returns:
        np.float64: accuracy
    """
    ### TODO Implement accuracy function
    n_total = len(y)
    accuracy =list((y==y_hat)).count(True)/n_total
    return accuracy
    
    
    raise NotImplementedError

# PSNR
def PSNR(img1, img2):
    
    return psnr(img1, img2)

def avg_PSNR(model, train_data : np.ndarray, train_labels : np.ndarray):
    return

# SSIM
def SSIM(img1, img2):
    return ssim(img1, img2, channel_axis = #TODO)

def avg_SSIM(model, train_data : np.ndarray, train_labels : np.ndarray):
    return #TODO

def approx_train_psnr_ssim(model, train_comp : np.ndarray, train_true : np.ndarray) -> np.float64:
    idxs = np.random.choice(len(train_data), 4000, replace=False)
    x = torch.from_numpy(train_comp[idxs].astype(np.float32))
    y = torch.from_numpy(train_true[idxs].astype(np.float32))
    restored = model(x)
    #TODO
    return accuracy(train_labels[idxs], y_pred.numpy()), loss.item()


def dev_loss_psnr_ssim((model, dev_comp : np.ndarray, dev_true : np.ndarray) -> np.float64:
    x = torch.from_numpy(dev_comp.astype(np.float32))
    y = torch.from_numpy(dev_true.astype(np.float32))
    restored = model(x)
    #loss = F.cross_entropy(restored, y)
    loss = F.l1_loss(restored, y)
    #TODO
    return loss.item(), dev_psnr, dev_ssim

