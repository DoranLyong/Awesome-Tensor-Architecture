# (ref) https://github.com/cs230-stanford/cs230-code-examples/blob/478e747b1c8bf57c6e2ce6b7ffd8068fe0287056/pytorch/vision/model/net.py


import numpy as np 
import torch 
import torch.nn as nn 


from .vgg import VGG



#%% 
class Net(nn.Module):

    def __init__(self, params):
        """ We define an convolutional network that predicts the sign from an image. 
        
            The components required are:
                - an embedding layer(e.g., VGG): this layer maps each image frame to a feature map 
                - fc: a fully connected layer that converts the VGG output to the final output 
        
            Args:
                params: (Params) contains 'dropout_rate' & 'num_classes' 
        """
        super(Net, self).__init__()

        # ===== embedding-layer 
        self.embedding = VGG(in_channels=3, type= 'VGG16').conv_layers 


        # ===== FC-layer 
        self.fc = nn.Sequential(nn.Linear(512*2*2, 4096),
                                nn.ReLU(),
                                nn.Dropout(p=params.dropout_rate),
                                nn.Linear(4096, 4096),
                                nn.ReLU(),
                                nn.Dropout(p=params.dropout_rate),
                                nn.Linear(4096, params.num_classes),
                                )          


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndimension() == 4, f"Input tensor to convolution must be 4d! but, {x.ndimension()}d tensor is given"

        x = self.embedding(x) # input := [1, 3, 64, 64]  =>  output := [1, 512, 2, 2]
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x 



