# VGG Network for Hand Signs Recognition 

This project format follows [here](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision).



### Requirements 

```bash
pip install -r requirements.txt
```



### Task 

Given an image of a hand doing a sign representing 0, 1, 2, 3, 4 or 5, predict the correct label.



### Download the SIGNS dataset

The dataset is hosted on google drive, download it [here](https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view).

This will download the SIGNS dataset (~1.1 GB) containing photos of hands signs making numbers between 0 and 5. Here is the structure of the data:

```bash 
SIGNS/
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
```

The images are named following `{label}_IMG_{id}.jpg` where the label is in `[0, 5]`. The training set contains 1,080 images and the test set contains 120 images.

Once the download is complete, move the dataset into `data/SIGNS`. Run the script `build_dataset.py` which will resize the images to size `(64, 64)`. The new resized dataset will be located by default in `data/64x64_SIGNS`:

```bash
python build_dataset.py --data_dir data/SIGNS --output_dir data/64x64_SIGNS
```



### Quickstart 

Check `run_tutorial.ipynb`.





***

### Reference 

[1] [VGG-tutorial, github](https://github.com/DoranLyong/VGG-tutorial/blob/main/VGG_pytorch/models.py) / <br/>
[2] [cs230-code-examples, github](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision) / 여기에 소개된 task를 VGG net으로 해결해보자 <br/>
[3] [involution.py, github](https://github.com/DoranLyong/Awesome_Tensor_Architecture/blob/main/Vision/tutorial01_Involution/models/involution.py) / 모듈 디자인은 여기 포멧을 따르기 <br/>
[4] [Intro to Pytorch Code Examples, CS230](https://cs230.stanford.edu/blog/pytorch/) / <br/>
[5] [torchvision, vgg.py](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py) / 

