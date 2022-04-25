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
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss)
from skimage.transform import rotate
from skimage.util import random_noise

class SupRes(nn.Module):
    ### TODO change #channels
    def __init__(self, upscale_factor = 2):
        super().__init__()
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=1)

        self.su1 = nn.Sequential(      
            nn.ReLU(),
            nn.Conv2d(in_channels=6,out_channels=6,kernel_size=1),                              
            nn.Sigmoid()                      
        )
 
        #Shrinking
        self.conv2=nn.Conv2d(in_channels=6,out_channels=6,kernel_size=1,stride=1,padding=0)
        self.su2 = nn.Sequential(      
            nn.ReLU(),   
            nn.Conv2d(in_channels=6,out_channels=6,kernel_size=1),                              
            nn.Sigmoid()                      
        )
 
        # Non-linear Mapping
        self.conv3=nn.Conv2d(in_channels=6,out_channels=6,kernel_size=3,stride=1,padding=1)
        self.su3 = nn.Sequential(      
            nn.ReLU(),   
            nn.Conv2d(in_channels=6,out_channels=6,kernel_size=1),                              
            nn.Sigmoid()                      
        )
        self.conv4=nn.Conv2d(in_channels=6,out_channels=6,kernel_size=3,stride=1,padding=1)
        self.su4 = nn.Sequential(      
            nn.ReLU(),   
            nn.Conv2d(in_channels=6,out_channels=6,kernel_size=1),                              
            nn.Sigmoid()                      
        )
 
 
        # Expanding
        self.conv5=nn.Conv2d(in_channels=6,out_channels=6,kernel_size=1,stride=1,padding=0)
        self.su5 = nn.Sequential(      
            nn.ReLU(),   
            nn.Conv2d(in_channels=6,out_channels=6,kernel_size=1),                              
            nn.Sigmoid()                      
        )
 
        # Deconvolution
        self.deconv= nn.ConvTranspose2d(in_channels=6,out_channels=3,kernel_size=7,stride=upscale_factor, padding=3, output_padding=1)

        # raise NotImplementedError
        
    
    def forward(self, x):
        
        ### TODO Implement your best model's forward pass module  
        #   
        x = self.conv1(x)
        x = torch.mul(self.su1(x), x)
        x = self.conv2(x)
        x = torch.mul(self.su2(x), x)
        x = self.conv3(x)
        x = torch.mul(self.su3(x), x)
        x = self.conv4(x)
        x = torch.mul(self.su4(x), x)
        x = self.conv5(x)
        x = torch.mul(self.su5(x), x)
        x = self.deconv(x)
        return x

        
        # raise NotImplementedError
    
    

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
        TRAIN_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_images.npy"))
        TRAIN_LABELS = np.load(os.path.join(DATA_DIR, "fruit_labels.npy"))
        DEV_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_dev_images.npy"))
        DEV_LABELS = np.load(os.path.join(DATA_DIR, "fruit_dev_labels.npy"))
        
        ### TODO format your dataset to the appropriate shape/dimensions necessary to be input into your model.
        ### TODO get the following parameters and name them accordingly: 
        # [N_IMAGES] Number of images in the training corpus 
        N_IMAGES = TRAIN_IMAGES.shape[0]
        # [HEIGHT] Height and [WIDTH] width dimensions of each image
        HEIGHT = TRAIN_IMAGES.shape[1]
        WIDTH = TRAIN_IMAGES.shape[2]
        # [N_CLASSES] number of output classes
        N_CLASSES = len(np.unique(TRAIN_LABELS))    
        

        # do online data augmentation via affine transformation
        rotate45 = rotate(TRAIN_IMAGES, 45)
        print(rotate45.shape)
        rotate90 = rotate(TRAIN_IMAGES, 90)
        rotate180 = rotate(TRAIN_IMAGES, 180)
        TRAIN_IMAGES = np.concatenate((TRAIN_IMAGES, rotate45, rotate90, rotate180))
        TRAIN_LABELS = np.concatenate((TRAIN_LABELS, TRAIN_LABELS, TRAIN_LABELS, TRAIN_LABELS))
        print(TRAIN_IMAGES.shape)

        ### TODO Normalize each of the individual images to a mean of 0 and a variance of 1
        # add a dimension of channels 
        flat_train_imgs = TRAIN_IMAGES[:, np.newaxis, :, :]
        flat_dev_imgs = DEV_IMAGES[:, np.newaxis, :, :]
        # Normalize 
        train_mean = flat_train_imgs.mean(axis=(2, 3), keepdims=True)
        train_std = flat_train_imgs.std(axis=(2, 3), keepdims=True)   
        flat_train_imgs = (flat_train_imgs - train_mean) / train_std
        dev_mean = flat_dev_imgs.mean(axis=(2, 3), keepdims=True)
        dev_std = flat_dev_imgs.std(axis=(2, 3), keepdims=True)   
        flat_dev_imgs = (flat_dev_imgs - dev_mean) / dev_std

        #TRAIN_LABELS = TRAIN_LABELS[:, np.newaxis]
        #DEV_LABELS = DEV_LABELS[:, np.newaxis]

        


        #raise NotImplementedError
        
        
        # do not touch the following 4 lines (these write logging model performance to an output file 
        # stored in LOG_DIR with the prefix being the time the model was trained.)
        LOGFILE = open(os.path.join(LOG_DIR, f"bestmodel.log"),'w')
        log_fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        ### TODO change depending on your model's instantiation
        
        #raise NotImplementedError
        model = BestModel(input_height = HEIGHT, input_width= WIDTH,
                                 n_classes=N_CLASSES)
        
        ### TODO (OPTIONAL) : you can change the choice of optimizer here if you wish.
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        
    
        for step in range(EPOCHS):
            i = np.random.choice(flat_train_imgs.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(flat_train_imgs[i].astype(np.float32))
            y = torch.from_numpy(TRAIN_LABELS[i].astype(np.int))
            
    

            
            # Forward pass: Get logits for x
            logits = model(x)
            
            # Compute loss
            loss = F.l1_loss(logits, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # log model performance every 100 epochs
            if step % 100 == 0:
                train_acc, train_loss = approx_train_acc_and_loss(model, flat_train_imgs, TRAIN_LABELS)
                dev_acc, dev_loss = dev_acc_and_loss(model, flat_dev_imgs, DEV_LABELS)
                step_metrics = {
                    'step': step, 
                    'train_loss': loss.item(), 
                    'train_acc': train_acc,
                    'dev_loss': dev_loss,
                    'dev_acc': dev_acc
                }

                print(f"On step {step}:\tTrain loss {train_loss}\t|\tDev acc is {dev_acc}")
                logger.writerow(step_metrics)
        LOGFILE.close()
        
        ### TODO (OPTIONAL) You can remove the date prefix if you don't want to save every model you train
        ### i.e. "{DATE_PREFIX}_bestmodel.pt" > "bestmodel.pt"
        model_savepath = os.path.join(MODEL_SAVE_DIR,f"{DATE_PREFIX}_bestmodel.pt")
        
        
        print("Training completed, saving model at {model_savepath}")
        torch.save(model, model_savepath)
        
        
    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None : raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None : raise TypeError("for inference, a predictions file must be specified for output.")
        TEST_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_test_images.npy"))
        
        model = torch.load(WEIGHTS_FILE)
        
        predictions = []
        for test_case in TEST_IMAGES:
            
            ### TODO implement any normalization schemes you need to apply to your test dataset before inference
            test_case = test_case[:, np.newaxis, :, :]
            test_mean = test_case.mean(axis=(2, 3), keepdims=True)
            test_std = test_case.std(axis=(2, 3), keepdims=True)   
            test_case = (test_case - test_mean) / test_std
            # raise NotImplementedError
            
            
            x = torch.from_numpy(test_case.astype(np.float32))
            x = x.view(1,-1)
            logits = model(x)
            pred = torch.max(logits, 1)[1]
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")

    else: raise Exception("Mode not recognized")