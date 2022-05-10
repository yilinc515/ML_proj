# List of models: 

## Main models
main_SupRes: initial model based on FSRCNN with selection unit modification. Underwent extensive hyperparameter tuning and testing.   
** main_SupResV1.1: version of our modified FSRCNN with best performance. Training results included in presentation. ** 
main_fsrcnnPReLUonly: main model 1.1 with PReLu activation as in FSRCNN (10000 epochs, 54 min)  
main_fsrcnn+res: main model 1.1 with residual (10000 epochs, around 55 min)   
main_fsrcnn+res4x: main model 1.1 with residual for 4x upscale  

## Benchmarking models
baseline: implements non-machine learning models including nearest neighbor, bilinear and bicubic interpolation
main_espcn1: simplest version of espcn with 3 convolutional layers (espcn1) (6000 epochs, 30 min)  
main_espcn2: simplest version of espcn with more channels (espcn2) (6000 epochs, 30 min)  
main_espcn(4layers): espcn with 4 convolutional layers, no selection unit  
main_selnet: simplified version of selnet (6000 epochs, 35 min; 20000 epochs, 1h45min)  
main_selnet4x: simplified version of selnet for 4x upscale (7500 epochs, 42 min, batch size=50)  






