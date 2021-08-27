# official     (ref) https://github.com/d-li14/involution/blob/main/det/mmdet/models/utils/involution_naive.py
# unofficial   (ref) https://github.com/shikishima-TasakiLab/Involution-PyTorch/blob/main/involution/involution2d.py
# tf_ver       (ref) https://github.com/ariG23498/involution-tf/blob/master/Involution.ipynb
# 2D, 3D ver   (ref) https://github.com/ChristophReich1996/Involution/blob/master/involution/involution.py
# keras ver    (ref) https://keras.io/examples/vision/involution/
#              (ref) https://github.com/eddie94/Deep-learning-utils/blob/main/involution/model.py
# tf-keras ver (ref) https://github.com/YirunKCL/Tensorflow-Keras-Involution2D/blob/main/involution2d.py
#%% 
from typing import (Union,     # multiple types are allowed ; (ref) https://www.daleseo.com/python-typing/
                    Tuple, 
                    Optional,  # type checker ; (ref) https://stackoverflow.com/questions/51710037/how-should-i-use-the-optional-type-hint
                    )

import torch 
import torch.nn as nn 



#%% 
class Involution2d(nn.Module):
    """ (ref) https://github.com/ChristophReich1996/Involution/blob/master/involution/involution.py
        (ref) https://arxiv.org/pdf/2103.06255.pdf
    """

    def __init__(self,  in_channels: int, 
                        out_channels: int, 
                        sigma_mapping: Optional[nn.Module] = None,         # (ref) https://www.daleseo.com/python-typing/
                        kernel_size: Union[int, Tuple[int, int]] = (7, 7), 
                        stride: Union[int, Tuple[int, int]] = (1, 1),
                        groups: int = 1,   # for grouped-convolution 
                        reduce_ratio: int = 1,
                        dilation: Union[int, Tuple[int, int]] = (1, 1),
                        padding: Union[int, Tuple[int, int]] = (3, 3),
                        bias: bool = False,   
                        **kwargs) -> None:
        """ [Constructor method]
            - in_channels: input channel number 
            - out_channels: output channel number
            - sigma_mapping: Non-linear mapping as introduced in the paper. If None, BN + ReLU is default
            - kernel_size: Kernel size to be used
            - stride: Stride factor to be utilized
            - groups: Number of groups to be employed
            - reduce_ratio: Reduce ration of involution channels
            - dilation: Dilation in unfold to be employed
            - padding: Padding to be used in unfold operation
            - bias: If true, bias is utilized in each convolution layer
            - **kwargs: Unused additional key word arguments
        """
        
        # ===== Call super constructor 
        super(Involution2d, self).__init__()

        # ===== Check parameters 
        assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."  
        assert in_channels % groups == 0, "out_channels must be divisible by groups"

        assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), "kernel size must be an int or a tuple of ints."
        assert isinstance(stride, int) or isinstance(stride, tuple), "stride must be an int or a tuple of ints."
        assert isinstance(groups, int), "groups must be a positive integer."
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."
        assert isinstance(dilation, int) or isinstance(dilation, tuple), "dilation must be an int or a tuple of ints."
        assert isinstance(padding, int) or isinstance(padding, tuple), "padding must be an int or a tuple of ints."
        assert isinstance(bias, bool), "bias must be a bool"

        # ===== save parameters 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) 
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.bias = bias

        # ===== Init. modules
        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential( 
                                            nn.BatchNorm2d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), # (ref) https://gaussian37.github.io/dl-concept-batchnorm/
                                            nn.ReLU(), # (ref) https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
                                            ) # Non-linear mapping    

        self.initial_mapping = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                         bias=bias) if self.in_channels != self.out_channels else nn.Identity() # (ref) https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
                                                                                                                # for kernel_generator 

        self.o_mapping = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)  # AvgPool2d vs. AdaptiveAvgPool2d 
                                                                                    # (ref) https://gaussian37.github.io/dl-pytorch-snippets/

        self.reduce_mapping = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels // self.reduce_ratio, 
                                        kernel_size=(1, 1),
                                        stride=(1, 1), 
                                        padding=(0, 0), 
                                        bias=bias
                                        )        

        self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
                                      out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups,
                                      kernel_size=(1, 1), 
                                      stride=(1, 1), 
                                      padding=(0, 0), 
                                      bias=bias
                                      )

        self.unfold = nn.Unfold(kernel_size=self.kernel_size,  # (ref) https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
                                dilation=dilation,             # (ref) https://npclinic3.tistory.com/6
                                padding=padding,               # watch this! ; (ref) https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
                                stride=stride
                                )


    def __repr__(self) -> str: # representation ; call with repr() 
        # __str__ vs. __repr__ ; https://shoark7.github.io/programming/python/difference-between-__repr__-vs-__str__
        """ This method returns information about this module
        """  
        print(f"==== {self.__class__.__name__} INFO ====")   

        return (f"in_channel, out_channel=({self.in_channels}, {self.out_channels}) \n"
                f"kernel_size=({self.kernel_size[0]}, {self.kernel_size[1]}) \n"
                f"stride=({self.stride[0]}, {self.stride[1]}) \n"
                f"padding=({self.padding[0]}, {self.padding[0]}) \n"
                f"groups={self.groups} \n" 
                f"reduce_ratio={self.reduce_mapping} \n" 
                f"dilation=({self.dilation[0]}, {self.dilation[1]}) \n"
                f"bias={self.bias} \n" 
                f"sigma_mapping={str(self.sigma_mapping)} \n"
                )          

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass 
            - input: Input tensor of the shape [batch_size, in_channels, height, width]
            - return: Output tensor of the shape [batch_size, out_channels, height, width] (w/ same padding)
        """                

        # ==== Check input dimension of input tensor
        # torch.tensor.ndimension ; (ref) https://pytorch.org/docs/stable/generated/torch.Tensor.ndimension.html
        assert input.ndimension() == 4, f"Input tensor to involution must be 4d! but, {input.ndimension()}d tensor is given"

        # ==== Save input shape and compute output shapes
        # (ref) https://justkode.kr/deep-learning/pytorch-cnn
        batch_size, _, in_height, in_width = input.shape # [B, C, H, W]
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1 
        out_width  = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)  // self.stride[1] + 1

        # ==== Unfold and reshape input tensor
        input_unfolded = self.unfold(self.initial_mapping(input)) # unfold ; (ref) https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work                                                                
        input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels // self.groups,
                                             self.kernel_size[0] * self.kernel_size[1],
                                             out_height, out_width) # reshape  ; https://hichoe95.tistory.com/26
                                                                    #          ; https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
        
        # a.k.a., Kernel_generator 
        kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))

        kernel = kernel.view(batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1],
                             kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)     

        # Apply kernel to produce output
        output = (kernel * input_unfolded).sum(dim=3)

        # Reshape output
        output = output.view(batch_size, -1, output.shape[-2], output.shape[-1])  # [batch_size, -1, height, width]
        return output                                                  



# %%



if __name__ == "__main__":
    # ===== Set device 
    gpu_no = 0
    device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # ====== Set test input 
    img = torch.randn(32, 3, 256, 256).to(device)


    # ====== Set model 
    # (ref) https://github.com/ChristophReich1996/Involution/blob/master/examples.py
    model = Involution2d(in_channels=3, out_channels=4).to(device)
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
