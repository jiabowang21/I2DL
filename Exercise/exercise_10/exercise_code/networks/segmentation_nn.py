"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        #self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        """
       
        self.cnn = nn.Sequential(

            nn.Conv2d(3, self.hparams['num_filters'], kernel_size = self.hparams['kernel_size'], padding = self.hparams['padding']),
            nn.ReLU(),

            nn.Conv2d(self.hparams['num_filters'], self.hparams['num_filters'] * 2, kernel_size = self.hparams['kernel_size'], padding = self.hparams['padding']),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.hparams['num_filters'] * 2, self.hparams['num_filters'] * 4, kernel_size = self.hparams['kernel_size'], padding = self.hparams['padding']),
            nn.ReLU(),

            nn.Conv2d(self.hparams['num_filters'] * 4, self.hparams['num_filters'] * 8, kernel_size = self.hparams['kernel_size'], padding = self.hparams['padding']),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.hparams['num_filters'] * 8, self.hparams['num_filters'], kernel_size = 1, padding = 0)
        )
        self.upsamling = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
        )
        self.adjust = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Conv2d(self.hparams['num_filters'], 23, kernel_size = 1, padding = 0)
        )
        """
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ELU()
        self.maxpool = nn.MaxPool2d(3, 3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        
        self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=8, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.convt3 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        
        self.up1=nn.Upsample(scale_factor=3, mode='nearest')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        """
        x = self.cnn(x)
        x = self.upsamling(x)
        x = self.adjust(x)
        """
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.size())
        
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.up1(x)
        
        #print(x.size())
        x = self.convt2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.up1(x)
        
        #print(x.size())
        x = self.convt3(x)
        x = self.relu(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
