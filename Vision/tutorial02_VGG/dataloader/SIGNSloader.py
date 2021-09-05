""" (ref) https://github.com/cs230-stanford/cs230-code-examples/blob/478e747b1c8bf57c6e2ce6b7ffd8068fe0287056/pytorch/vision/model/data_loader.py#L60
    (ref) http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    (ref) http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

"""  define a training image loader that specifies transforms on images.
"""

import random
import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms



# =============== #
#    Transforms   #
# =============== #
train_transformer = transforms.Compose([transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
                                        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
                                        transforms.ToTensor(), # transform it into a torch tensor
                                        ])

eval_transformer = transforms.Compose([ transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
                                        transforms.ToTensor(),  # transform it into a torch tensor                                        
                                        ])  




# =================== #
#    Dataset loader   #
# =================== #
class SIGNSDataset(Dataset):
    """ A standard PyTorch definition of Dataset 
        which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir:str, transform:torchvision.transforms):
        """ Store the filenames of the jpgs to use. 
            Specifies transforms to apply on images.

            Args: 
                data_dir: directory path containing the dataset
                transform: transformation to apply on image
        """
        
        self.filenames = os.listdir(data_dir)
        self.filePaths = [osp.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(osp.split(filepath)[-1][0]) for filepath in self.filePaths]  # '0_IMG_5942.jpg' -> '0'
        self.transform = transform

    def __len__(self) -> int:
        # return size of dataset 
        return len(self.filePaths)

    def __getitem__(self, idx:int):
        """ Fetch index idx image and labels from dataset. 
            Perform transforms on image.

            Args: 
                idx: index in [0, 1, ..., size_of_dataset-1]

            Returns: 
                image: (Tensor) transformed image
                label: (int) corresponding label of image
        """
        image = Image.open(self.filePaths[idx])  # PIL image
        image = self.transform(image)

        return image, self.labels[idx]



# ==================== #
#    Fetch dataloader  #
# ==================== #
def fetch_dataloader(types:list, data_dir:str, params) -> dict:
    """ Fetches the DataLoader object for each type in types from data_dir.

        Args: 
            types: has one or more of ['train', 'val', 'test'] depending on which data is required
            data_dir: directory path containing the dataset
            params: (utils.Params) hyperparameters

        Returns:
            data: contains the DataLoader object for each type in types
    """
    dataloaders = {}

    data_types = ['train', 'val', 'test']
    for split in data_types:

        if split in types: 
            path = osp.join(data_dir, f"{split}_signs")

            # use the 'train_transformer' if training data, 
            # else use 'eval_transformer' without random flip
            if split == 'train':

                dl = DataLoader(SIGNSDataset(path, train_transformer), 
                                batch_size=params.batch_size, 
                                shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda, # True? False? (ref) https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
                                )
            else: 
                dl = DataLoader(SIGNSDataset(path, eval_transformer), 
                                batch_size=params.batch_size, 
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda,
                                )
            
            dataloaders[split] = dl

    return dataloaders
