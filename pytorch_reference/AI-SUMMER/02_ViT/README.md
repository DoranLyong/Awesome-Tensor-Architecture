# Vision Transformer (ViT) Tutorial 

※ refer to [AI SUMMER](https://theaisummer.com/hugging-face-vit/) 



Transformers lack(없다) the ```inductive biases``` of Convolutional Neural Networks (```CNN```s), such as:

* ```translation invariance``` 
* and a locally restricted [receptive field](https://theaisummer.com/receptive-field/). 

__Translation__ : in computer vision it implies that each image pixel has been moved by a fixed amount in a particular direction. __Invariance__ : it means that you can recognize an entity (i.e. object) in an image, even when its appearance or position varies.

Moreover, remember that ```convolution``` is a ```linear local operator```. We see only the neighbor values as indicated by the ```kernel```. 



On the other hand, the ```transformer``` is by design **permutation invariant**. The bad news is that it ```cannot process``` ```grid-structured data```. We need ```sequences```. We will convert a ```spatial non-sequential signal``` to a ```sequence```!



## How the Vision Transformer works?

<img src="./page_imgs/ViT.gif" width="640">

[Vision Transformer architecture](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html) :

1. Split an ```image``` into ```patches``` 

2. ```Flatten``` the patches 

3. Produce ```lower-dimensional linear embeddings``` from the flattened patches 

4. Add ```positional embeddings```   

   (__Sequentialization done__!)

5. Feed the sequence as an input to a standard transformer encoder 

6. ```Pretrain``` the model with image labels (```fully supervised``` on a huge dataset)

7. ```Finetune``` on the downstream dataset for image classification



```Image patches``` are basically the ```sequence tokens``` (like words). In fact, the encoder block is identical to the original transformer proposed by Vaswani et al. (2017) as [described](https://theaisummer.com/transformer/):

![the-transformer-block-vit](https://theaisummer.com/static/aa65d942973255da238052d8cdfa4fcd/7d4ec/the-transformer-block-vit.png)

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, 2020](https://arxiv.org/abs/2010.11929)

