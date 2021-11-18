import os 
import os.path as osp 

import numpy as np 
import pandas as pd
from PIL import Image

import torch 
from torch.utils import data 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T 


# === Transforms === # 

train_transforms = T.Compose([  T.RandomCrop(32, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                            std=(0.2023, 0.1994, 0.2010))
                                ])

test_transforms = T.Compose([   T.ToTensor(),
                                T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                            std=(0.2023, 0.1994, 0.2010))
                                ])



# === Fetch dataloader === # 
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

                dl = DataLoader(ImageDataset(path, train_transforms), 
                                batch_size=params.batch_size, 
                                shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda, # True? False? (ref) https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
                                )
            else: 
                dl = DataLoader(ImageDataset(path, test_transforms), 
                                batch_size=params.batch_size, 
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda,
                                )
            
            dataloaders[split] = dl

    return dataloaders






# === Dataset Definition === # 
class ImageDataset(data.Dataset):
    def __init__(self, data_dir:str, transform=None):
        super(ImageDataset, self).__init__()

        self.filenames = os.listdir(data_dir)
        self.filePaths = [osp.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(osp.split(filepath)[-1][0]) for filepath in self.filePaths]  # '0_IMG_5942.jpg' -> '0'

        self.transform = transform        


    def __len__(self) -> int:
        # return size of dataset
        len(self.filePaths)


    def __getitem__(self, idx:int):
        """ Fetch index idx image and labels from dataset. 
            Perform transforms on image.

            Args: 
                idx: index in [0, 1, ..., size_of_dataset-1]
            
            Returns: 
                image: (Tensor) transformed image
                label: (int) corresponding label of image
        """

        image = Image.open(self.filePaths[idx])
        
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]




