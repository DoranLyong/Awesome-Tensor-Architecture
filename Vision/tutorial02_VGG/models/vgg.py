# VGG design ; (ref) https://github.com/DoranLyong/VGG-tutorial/blob/main/VGG_pytorch/models.py 
# code format ; (ref) https://github.com/DoranLyong/Awesome_Tensor_Architecture/blob/main/Vision/tutorial01_involution/models/involution.py
#%%
from typing import (Union,     # multiple types are allowed ; (ref) https://www.daleseo.com/python-typing/
                    Tuple, 
                    Optional,  # type checker ; (ref) https://stackoverflow.com/questions/51710037/how-should-i-use-the-optional-type-hint
                    )

import torch 
import torch.nn as nn 
import torch.nn.functional as F 



#%% 
class VGG(nn.Module):

    VGG_types = {   'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
                }


    def __init__(self, params,  num_classes:int, type:str = 'VGG16'):
        super(VGG, self).__init__() 

        # ===== Conv-layer 
#        self.in_channels = params.num_channels 
        self.in_channels = 3
        self.conv_layers = self.create_conv_layers(self.VGG_types[type])


        # ===== Dense 
        self.dense = nn.Sequential( nn.Linear(512*7*7, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, num_classes),
                                )        

    def create_conv_layers(self, architecture:list) -> nn.Module:
        """ VGG-n options 
        """
        layers = [] 
        in_channels = self.in_channels 

        for x in architecture:
            if type(x) == int: 
                out_channels = x 

                layers += [ nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                            nn.BatchNorm2d(x),
                            nn.ReLU(),
                            ]

                in_channels = x   # present out_channels will be the next in_channels          

            elif x == 'M':
                """ MaxPooling
                """      
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]           

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndimension() == 4, f"Input tensor to involution must be 4d! but, {x.ndimension()}d tensor is given"

        x = self.conv_layers(x) # input := [1, 3, 224, 224]  =>  output := [1, 512, 7, 7]

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dense(x)
        return x 



#%% 
if __name__ == "__main__":
    # ===== Set device 
    gpu_no = 0
    device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # ====== Set test input 
    img = torch.randn(32, 3, 224, 224).to(device)    

    # ====== Set model 
    model = VGG(3,  num_classes=10, type = 'VGG16').to(device)
    print(model)  # by __repr__ method     


    # ====== Get parameters 
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count += 1
            print(name) # trainable layer 

    print(f"# of trainable layer : {count} \n")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # (ref) https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    total_params = sum(p.numel() for p in model.parameters()) # torch.numel() ; (ref) https://pytorch.org/docs/stable/generated/torch.numel.html
    print(f"# of trainable params: {trainable_params}")
    print(f"# of params: {total_params}")


    # ====== output test 
    out = model(img)
    print(out.shape)