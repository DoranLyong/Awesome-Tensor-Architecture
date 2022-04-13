# 작물 잎 사진으로 질병 분류하기 

## Contents 

* [데이터 준비](#데이터 준비)
* [데이터 구조](#데이터 구조)
* [실험 설계를 위한 데이터 분할](#실험 설계를 위한 데이터 분할)
* [데이터셋 및 데이터 로더(dataloader) 생성](#데이터셋 및 데이터 로더(dataloader) 생성)
* [베이스라인 모델 설계](#베이스라인 모델 설계)
* [전이 학습 (Transfer Learning)](#전이 학습 (Transfer Learning))



## 데이터 준비 

데이터 출처: [bjpublic's github](https://github.com/bjpublic/DeepLearningProject/tree/main/04_%EC%9E%91%EB%AC%BC_%EC%9E%8E_%EC%82%AC%EC%A7%84_%EC%A7%88%EB%B3%91_%EB%B6%84%EB%A5%98)

* 위의 링크로 이동하여 ```dataset.zip``` 파일을 다운 받고 압축을 해제합니다. 

```yaml
# 프로젝트 구조 
/
├ dataset     # raw dataset 
├ processsed  # 전치리로 분할된 dataset 
└ Plant-Leaf-Classification.ipynb  # 프로젝트 코드 
```



## 데이터 구조 

각 이미지의 분류 클래스가 각각 디렉토리로 구분되어 있는 형태. 

* Train, Validation, Test 데이터가 따로 구별되어 있지 않음. 
* 클래스 레이블은 디렉토리 이름으로 대체 

<img src="./imgs/data_structure.png" width=640> 

```yaml
dataset
├ Apple___Apple_scab 
│	├ image (1).JPG
│	├ ...
│	└ image (630).JPG
... 
└ Tomato___Tomato_Yellow_Leaf_Curl_Virus
	├ image (1).JPG
	├ ...
	└ image (5357).JPG
```



## 실험 설계를 위한 데이터 분할 

위의 raw dataset 을 학습을 위해 Train, Validation, Test dataset으로 분할하는 전처리 진행. 

* train, val, test 디렉토리를 생성 
* 각 디렉토리 하위로 클래스의 목록에 해당하는 디렉토리를 추가로 생성
* 마지막으로 ```dataset``` 으로 부터 이미지 데이터를 복사해옴 (train:val:test = 6:2:2)

<img src="./imgs/data_split.png" width=480> 

```yaml
processed 
├ train
│	├ Apple___Apple_scab 
│	├ ... 
│	└ Tomato___Tomato_Yellow_Leaf_Curl_Virus
├ val 
│	├ Apple___Apple_scab 
│	├ ... 
│	└ Tomato___Tomato_Yellow_Leaf_Curl_Virus
└ test 
	├ Apple___Apple_scab 
	├ ... 
	└ Tomato___Tomato_Yellow_Leaf_Curl_Virus
```



## 데이터셋 및 데이터 로더(dataloader) 생성 

필요한 개념과 기능: 

* ```torchvisiondatasets.ImageFolder```  메소드 
  * 스토리지(storage)상의 데이터셋을 PyTorch 환경으로 불러오는 메소드
  * 여기서 사용되는 dataset은 ```하나의 클래스가 하나의 디렉토리에 대응 됨```. 
    * 이러한 구조의 데이터셋을 로드할 때는 ```ImageFolder``` 메소드를 사용하면 간단함
    * (왜?) 디렉토리별 label encoding 을 해당 메소드가 내부적으로 해주기 때문

* ```torch.utils.data.DataLoader``` 

  * PyTorch 환경으로 불러온 dataset을 주어진 조건에 따라 미니 배치 단위로 분리하는 역할 수행 

  

## 베이스라인 모델 설계 

필요한 개념과 기능: 

* 



## 전이 학습 (Transfer Learning)

<img src="./imgs/fine-tuning-strategy.png" width=640> 

[데이터크기-유사성 그래프 & 각 상황에 따른 Fine-tuning 전략 (source)](https://jeinalog.tistory.com/13)
