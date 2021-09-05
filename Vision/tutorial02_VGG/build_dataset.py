# (ref) https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/build_dataset.py
"""
Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and val sets.
Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take 20% of "train_signs" as val set.
"""
import argparse 
import random 
import os 
import os.path as osp 
from pathlib import Path 
from collections import OrderedDict

from tqdm import tqdm 
from PIL import Image 


SIZE = 64
random.seed(230)


# =============== # 
# Argument by CLI # 
# =============== # 
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SIGNS', help="Directory with the SIGNS dataset")  # to read 
parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new data")  # to save 



# =============== # 
#  Resize & Save  # 
# =============== # 
def resize_and_save(filename:str, output_dir:str, size:int=SIZE):
    """ - Resize the image contained in `filename` 
        - Save it to the `output_dir`
    """
    image = Image.open(filename)

    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(osp.join(output_dir, filename.split('/')[-1])) # get file name 




if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from CLI 

    assert osp.isdir(args.data_dir), f"Couldn't find the dataset @ {args.data_dir}"


    # ===== Define the data dir. 
    train_data_dir = osp.join(args.data_dir, 'train_signs')
    test_data_dir = osp.join(args.data_dir, 'test_signs')


    # ===== Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir) 
    filePaths = [osp.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filePaths = [osp.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]



    # ===== Split the images in 'train_signs' into 80% train and 20% val
    #       Make sure to always shuffle with a fixed seed so that the split is reproducible
    filePaths.sort()
    random.shuffle(filePaths)

    split = int(0.8 * len(filePaths))   # split into 8:2
    train_filePaths = filePaths[:split]
    val_filePaths = filePaths[split:]


    items =[('train', train_filePaths),
            ('val', val_filePaths),
            ('test', test_filePaths),]
            
    filePaths_dict = OrderedDict(items) # (ref) https://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain/25480206
    
    

    # ===== Make a directory to save 
    jpg_saveDIR = Path(args.output_dir)
    jpg_saveDIR.mkdir(parents=True, exist_ok=True) 



    # ===== Spliting preprocess train, val and test
    for key in filePaths_dict.keys():
        output_dir_split = osp.join(args.output_dir, f'{key}_signs')
        split_saveDIR = Path(output_dir_split)
        split_saveDIR.mkdir(parents=True, exist_ok=True) 

        print(f"Processing {key} data, saving preprocessed data to {output_dir_split}")

        for path in tqdm(filePaths_dict[key]):
            resize_and_save(path, output_dir_split, size=SIZE)

    print("Done building dataset")

