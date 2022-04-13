# 작물 잎 사진으로 질병 분류하기 



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
  * 스토리지(storage)상의 데이터셋을 python 환경으로 불러오는 메소드





## 베이스라인 모델 설계 

