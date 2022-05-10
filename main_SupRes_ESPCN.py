# -*- coding: utf-8 -*-
"""
Created on Sat May  7 22:57:29 2022

@author: Lenovo
"""

import sys
import csv
import os
from turtle import xcor
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (approx_train_psnr_ssim, dev_loss_psnr_ssim)
from skimage.transform import rotate
#from skimage.util import random_noise

class SupRes(nn.Module):
    ### TODO change #channels
    def __init__(self, upscale_factor = 2):
        super().__init__()
        
        #the three convolutional layers
        self.conv1=nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, 
                             padding=1)
        
        self.conv2=nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, 
                             padding=1)
        
        self.conv3=nn.Conv2d(in_channels=8, out_channels=3*upscale_factor*upscale_factor, 
                             kernel_size=3, stride=1, padding=1)
 
                       
 
        #pixelshuffle
        self.pixelshuffle=nn.PixelShuffle(upscale_factor)
        #self.sigmoid = nn.Sigmoid()

        # raise NotImplementedError
        
    
    def forward(self, x):
        
        ### TODO Implement your best model's forward pass module  
        #   
        x = nn.ReLU(self.conv1(x))
        x = nn.ReLU(self.conv2(x))
        x = self.conv3(x)
        x = self.pixelshuffle(x)
        #x = self.Sigmoid(x)

        return x
    
    

if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')
    
    
    if MODE == "train":
        
        LOG_DIR = arguments.get('log_dir')
        MODEL_SAVE_DIR = arguments.get('model_save_dir')
        LEARNING_RATE = arguments.get('lr')
        BATCH_SIZE = arguments.get('bs')
        EPOCHS = arguments.get('epochs')
        DATE_PREFIX = datetime.datetime.now().strftime('%Y%m%d%H%M')
        if LEARNING_RATE is None: raise TypeError("Learning rate has to be provided for train mode")
        if BATCH_SIZE is None: raise TypeError("batch size has to be provided for train mode")
        if EPOCHS is None: raise TypeError("number of epochs has to be provided for train mode")
        TRAIN_COMP_IMAGES = np.load(os.path.join(DATA_DIR, "train_comp_images.npy"))
        TRAIN_TRUE_IMAGES = np.load(os.path.join(DATA_DIR, "train_true_images.npy"))
        DEV_COMP_IMAGES = np.load(os.path.join(DATA_DIR, "dev_comp_images.npy"))
        DEV_TRUE_IMAGES = np.load(os.path.join(DATA_DIR, "dev_true_images.npy"))
        
        ### TODO format your dataset to the appropriate shape/dimensions necessary to be input into your model.
        ### TODO get the following parameters and name them accordingly: 
        # [N_IMAGES] Number of images in the training corpus 
        N_IMAGES = TRAIN_COMP_IMAGES.shape[0]
        # [HEIGHT] Height and [WIDTH] width dimensions of each image
        HEIGHT = TRAIN_COMP_IMAGES.shape[1]
        WIDTH = TRAIN_COMP_IMAGES.shape[2]
  
    
        # do online data augmentation via affine transformation
        # rotate45 = rotate(TRAIN_COMP_IMAGES, 45)
  
        # rotate90 = rotate(TRAIN_COMP_IMAGES, 90)
        # rotate180 = rotate(TRAIN_COMP_IMAGES, 180)
        # TRAIN_IMAGES = np.concatenate((TRAIN_COMP_IMAGES, rotate45, rotate90, rotate180))
        # TRAIN_TRUE_IMAGES = np.concatenate(rotate(TRAIN_TRUE_IMAGES, 45), rotate(TRAIN_TRUE_IMAGES, 90), rotate(TRAIN_TRUE_IMAGES, 180))

        ### TODO Normalize each of the individual images to a mean of 0 and a variance of 1
        # add a dimension of channels 
        #flat_train_imgs = TRAIN_IMAGES[:, np.newaxis, :, :]
        #flat_dev_imgs = DEV_IMAGES[:, np.newaxis, :, :]

        # # Normalize 
        # train_mean = flat_train_imgs.mean(axis=(2, 3), keepdims=True)
        # train_std = flat_train_imgs.std(axis=(2, 3), keepdims=True)   
        # flat_train_imgs = (flat_train_imgs - train_mean) / train_std
        # dev_mean = flat_dev_imgs.mean(axis=(2, 3), keepdims=True)
        # dev_std = flat_dev_imgs.std(axis=(2, 3), keepdims=True)   
        # flat_dev_imgs = (flat_dev_imgs - dev_mean) / dev_std

        #TRAIN_LABELS = TRAIN_LABELS[:, np.newaxis]
        #DEV_LABELS = DEV_LABELS[:, np.newaxis]

        


        #raise NotImplementedError
        
        
        # do not touch the following 4 lines (these write logging model performance to an output file 
        # stored in LOG_DIR with the prefix being the time the model was trained.)
        LOGFILE = open(os.path.join(LOG_DIR, f"SupRes.log"),'w')
        log_fieldnames = ['step', 'train_loss','train_PSNR', 'train_SSIM', 'dev_loss','dev_PSNR', 'dev_SSIM']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        ### TODO change depending on your model's instantiation
        
        #raise NotImplementedError

        model = SupRes(upscale_factor = 2)
        
        ### TODO (OPTIONAL) : you can change the choice of optimizer here if you wish.
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        
    
        for step in range(EPOCHS):
            i = np.random.choice(TRAIN_COMP_IMAGES.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(TRAIN_COMP_IMAGES[i].astype(np.float32))
            y = torch.from_numpy(TRAIN_TRUE_IMAGES[i].astype(np.float32))

            
            # Forward pass: Get restored  image for x
            restored = model(x)
            
            # Compute loss
            loss = F.l1_loss(restored, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # log model performance every  epochs
            if step % 50 == 0:
                train_psnr, train_ssim = approx_train_psnr_ssim(model, TRAIN_COMP_IMAGES, TRAIN_TRUE_IMAGES)
                dev_loss, dev_psnr, dev_ssim = dev_loss_psnr_ssim(model, DEV_COMP_IMAGES, DEV_TRUE_IMAGES)
                step_metrics = {
                    'step': step, 
                    'train_loss': loss.item(), 
                    'train_PSNR': train_psnr,
                    'train_SSIM': train_ssim,
                    'dev_loss': dev_loss,
                    'train_PSNR':dev_psnr, 
                    'train_SSIM':dev_ssim
                }

                print(f"On step {step}:\tTrain loss {loss.item()}\tTrain PSNR {train_psnr}\tTrain SSIM {train_ssim}\tDev loss {dev_loss}\tDev PSNR {dev_psnr}\tDev SSIM {dev_ssim}")
                logger.writerow(step_metrics)
        LOGFILE.close()
        
        ### TODO (OPTIONAL) You can remove the date prefix if you don't want to save every model you train
        ### i.e. "{DATE_PREFIX}_bestmodel.pt" > "bestmodel.pt"
        model_savepath = os.path.join(MODEL_SAVE_DIR,f"{DATE_PREFIX}_supres.pt")
        
        
        print("Training completed, saving model at {model_savepath}")
        torch.save(model, model_savepath)
        
        
    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None : raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None : raise TypeError("for inference, a predictions file must be specified for output.")
        TEST_IMAGES = np.load(os.path.join(DATA_DIR, "dev_comp_images.npy"))
        
        model = torch.load(WEIGHTS_FILE)
        
        predictions = []
        for test_case in TEST_IMAGES:
            
            ### TODO implement any normalization schemes you need to apply to your test dataset before inference
            # test_case = test_case[:, np.newaxis, :, :]
            # test_mean = test_case.mean(axis=(2, 3), keepdims=True)
            # test_std = test_case.std(axis=(2, 3), keepdims=True)   
            # test_case = (test_case - test_mean) / test_std
            # raise NotImplementedError
            
            x = torch.from_numpy(test_case.astype(np.float32))
            #x = x.view(1,-1)
            pred = model(x)
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.save(PREDICTIONS_FILE, predictions)

    else: raise Exception("Mode not recognized")