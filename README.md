Hyperparameters: epochs: 1500 learning rate: 0.01 batch size: 200

Network architecture: 
Conv Layer 1: 12 channels
ReLU
Conv Layer 2: 24 channels
ReLU
Dropout
MaxPool
Linear Layer 1: 50 nodes
ReLU
Dropout
Final Linear Layer: 6 classes

Other methods tried:
Full CNN without dense layer: did not work, too many layers and channels does not work well with this dataset
AvgPool between conv layers: did not work
Data augmentation via affine transformation with scale, rotation and shear: did not work, online augmentation makes batch size too large, unsatisfactory accuracy

Batchsize > 400 and learning rate > 0.05  does not work well

What seems to work:
combination of convoluted layers and dense layers
smaller model with less layers and hidden nodes/channels


