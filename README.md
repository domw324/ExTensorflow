# ExTensorflow
Tensorflow 공부해보자

# 설치
설치 참고 사이트
- https://sdc-james.gitbook.io/onebook/
- https://pythonkim.tistory.com/137
- https://twinstarinfo.blogspot.com/2018/12/tensorflow-gpu-install-nvidia-cuda.html

## 1. Anaconda 설치
- 아나콘다는 ~

## 2. Tensorflow 설치
- 텐서플로 ~

## 3. Keras 설치
- 케라스는 ~

## 4. PyCharm 설치
- 파이참은 ~

# Tensorflow
공부하는 방법...
 
# Keras
Tensorflow 2.0에서는 Keras를 사용하길 권장하는 것 같다.

#### 참고사이트
아래 사이트를 따라가며 공부합니다. 이 글의 내용은 학습용으로 필기한 것이며, 해당 내용은 아래 블로그에서 가져왔습니다.
- 김태영의 케라스 블로그 : https://tykimos.github.io/index.html

## 딥러닝 개념잡기
### DataSet
- 딥러닝 모델을 학습시키기 위한 데이터. 풀고자 하는 문제 및 만들고자 하는 모델에 따라 데이터셋 설계도 달라진다.
- 훈련셋을 이용해 모델을 학습하고 가중치를 갱신하며, 검증셋으로 모델을 평가합니다. 검증셋으로 가장 높은 평가를 받은 학습 방법이 최적의 학습 방법이다.
- 학습방법을 결정하는 파라미터를 **하이퍼파라미터(hyperparameter)**라고 하고, 최적의 학습방법을 찾아가는 것을 하이퍼파라미터 튜닝이라고 한다.

1. 훈련셋
    - 모델 학습을 위한 데이터셋
    - input 값으로 얻은 출력값이 원하는 값과 같은지 비교하며 학습 한다.
    - **에포크(epochs)** : 학습 반복 횟수
    - **언더피팅(underfitting)**: 학습을 더 하면 모델의 성능이 높아질 가능성이 있는 상태. 초기 에포크 횟수가 경우
    - **오버피팅(overfitting)** : 학습이 더이상 모델의 성능을 높이지 않는 상태.
    - **조기종료(early stopping)** : 언더피팅~오버피팅이 되는 때 충분히 학습이 되었으므로, 학습 중단.
    - **가중치(Weight)** 갱신
2. 검증셋
    - 얼마 정도의 반복 학습이 좋은지 정하기 위한 데이터셋. 모델 검증 및 튜닝
    - 학습 중단 시점을 정할 수 있음
    - **가중치(Weight)** 갱신하지 않음
3. 시험셋
    - 학습이 완료된 후에 모델의 최종 성능 평가를 위한 데이터셋
    - **가중치(Weight)** 갱신하지 않음
    - 학습 과정에 전혀 관여하지 않음
    
### 학습과정
#### 학습
케라스에서 만든 모델을 학습할 때는 fit()함수 사용
```python
from keras.models import Sequential

model = Sequential() # model 생성

# ... model 세팅 ...

model.fit(x, y, batch_size=32, epochs=10)

# x : 입력 데이터
# y : 라벨 값
# batch_size : 가중치를 갱신할 샘플 개수 (데이터 셋은 다수의 샘플로 이루어짐)
# epochs : 학습 반복 횟수
```
- 배치 사이즈가 작을수록 가중치 갱신이 자주 일어남. 학습율은 높아지고, 하나의 데이터셋 학습에 걸리는 시간은 많이 걸림
- 배치 사이즈가 클수록. 학습율은 낮지만 하나의 데이터셋 학습에 걸리는 시간은 적게 걸림

#### 조기 종료
- 에포크가 커지면 모델의 성능은 높아지지만, 일정 수준 이상으로 커지면 **오버피팅(overfitting)** 발생
- 에포크가 늘어나면 value_loss가 감소, 에포크가 일정 수준 이상으로 커지면 **과적합**이 발생하면서 value_loss가 증가함
- 성능 개선의 여지가 없을 때 학습을 종료시키는 콜백함수를 호출.
```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
# moniter : 관찰하고자 하는 항목. 주로 val_loss, val_acc 사용
# min_delta : 개선되고 있다고 판단하기 위한 최소 변화량. 변화량 < min_delta이면 개선 없다고 판단
# patience : 개선이 없어도 종료하지 않고 개선 없는 에포크를 얼마나 기다려 줄 것인가 지정.
# verbose : 얼마나 자세하게 정보를 표시할 것인가 지정 (0, 1, 2)
# mode : 관찰 항목에 개선이 없다고 판단하기 위한 기준
#       ㄴauto : 관찰 항목의 이름에 따라 자동으로 지정
#       ㄴmin : 관찰 항목의 감소가 멈추면 종료
#       ㄴmax : 관찰 항목의 증가가 멈추면 종료

# 사용 예
early_stopping = EarlyStopping()
model.fit(X_train, Y_train, nb_epoch= 1000, callbacks=[early_stopping])
```
- 조기 종료를 하는 것은 좋지만, 어떤 모델은 손실값이 늘었다가 다시 줄어드는 경우가 발생할 수 있음. 즉, 좀 더 학습 될 수 있는 상태. 이 경우 **성급한 조기종료**가 발생한다.
- 성급한 조기종료를 방지하기 위해서 콜백함수에 patience 인자를 사용할 수 있다.  
```python
early_stopping = EarlyStopping(patience = 20) # 조기 종료 할 수 있어도 20 epochs 대기
model.fit(X_train, Y_train, nb_epoch= 1000, callbacks=[early_stopping])
```

#### 평가 : 분류
모델에 따라 적합한 평가 기준을 선택하는 것도 중요하다. 학습한 모델을 평가하는 방법으로는 **정확도**, **민감도**, **특이도**, **재현율** 등이 있다.
##### 정확도, 민감도, 특이도
- 정확도
    - 양성을 양성, 음성을 음성으로 분류하는 개수의 비율
    - 정확도 평가 시 클래스의 분포도를 꼭 확인해야 한다.
- 민감도
    - 양성에 민감한 정도
    - 양성을 양성으로 판정을 잘 할수록 민감도가 높다.
    - 민감도 = 판정한 것 중 실제 양성 수 / 전체 양성 수
- 특이도
    - 음성에 민감한 정도
    - 음성을 음성으로 판정을 잘 할수록 특이도가 높다.
    - 특이도 = 판정한 것 중 실제 음성 수 / 전체 음성 수
    
##### 임계값 (threshold)
양성 혹은 음성으로 판정하기 위한 기준. 임계값 조정을 통해 각기 다른 평가값을 얻게 된다.
- ROC(Receiver Operating Characteristic) curve : 민감도와 특이도가 어떤 관계를 가지고 변하는지를 나타낸 그래프
- AUC(Area Under Curve) : ROC curve의 아래면적을 구한 값. 쉽게 성능 비교를 할 수 있다.
- **sklearn** 패키지를 이용하면 ROC curve를 그릴 수 있고 AUC 값을 알려준다.
```python
# 사용 예
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class_A = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
proba_A = np.array([0.05, 0.15, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.95, 0.95])

class_B = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
proba_B = np.array([0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.65, 0.75, 0.85, 0.95])

false_positive_rate_A, true_positive_rate_A, thresholds_A = roc_curve(class_A, proba_A)
false_positive_rate_B, true_positive_rate_B, thresholds_B = roc_curve(class_B, proba_B)
roc_auc_A = auc(false_positive_rate_A, true_positive_rate_A)
roc_auc_B = auc(false_positive_rate_B, true_positive_rate_B)

plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')

plt.plot(false_positive_rate_A, true_positive_rate_A, 'b', label='Model A (AUC = %0.2f)'% roc_auc_A)
plt.plot(false_positive_rate_B, true_positive_rate_B, 'g', label='Model B (AUC = %0.2f)'% roc_auc_B)
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')
plt.show()
```
#### 평가 : 검출 및 검색
검출 문제에서는 분류문제와 다르게 검출되지 않은 진짜 음성에 대해서는 관심이 없다.
##### 정밀도, 재현율
- 정밀도
    - 모델이 얼마나 정밀한가
    - 양성만을 잘 고를 수록 정밀도가 높다.
    - 정밀도 = 실제 양성 수 / 양성이라고 판정한 수
- 재현율
    - 양성인 것을 놓치지 않는 비율
    - 양성을 많이 고를수록 재현율이 높다.
    - 재현율 = 검출 양성 수 / 전체 양성 수
    
##### 임계값 (threshold)
역시 임계값 조정을 통해 검출값이 달라진다.
- Precision-Recall Graph : 임계값에 따라 검출값이 달라지는 패턴을 볼 수 있는 그래프
- AP(Average Precision) : Precision-Recall Graph를 하나의 수치로 나타낸 것. 각 재현율에 해당하는 정밀도를 더해 평균을 취함.
- 역시 **sklearn**패키지를 이용해 구할 수 있다.
```python
# 사용 예
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

class_A = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1])
proba_A = np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.35, 0.35, 0.45, 0.45, 0.55, 0.55, 0.65, 0.85, 0.95])

class_B = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
proba_B = np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.25, 0.35, 0.35, 0.45, 0.55, 0.55, 0.65, 0.75, 0.95])

precision_A, recall_A, _ = precision_recall_curve(class_A, proba_A)
precision_B, recall_B, _ = precision_recall_curve(class_B, proba_B)

ap_A = average_precision_score(class_A, proba_A)
ap_B = average_precision_score(class_B, proba_B)

plt.title('Precision-Recall Graph')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.plot(recall_A, precision_A, 'b', label = 'Model A (AP = %0.2F)'%ap_A)   
plt.plot(recall_B, precision_B, 'g', label = 'Model B (AP = %0.2F)'%ap_B)  

plt.legend(loc='upper right')
plt.show()
```

#### 평가 : 분할
##### 픽셀 정확도, 평균 정확도, MeanIU
- 픽셀 정확도 (Pixel Accuracy)
    - 주로 이미지 처리에서 나오는 개념
    - 픽셀정확도 = (A 적중 수 + B 적중 수 + ...) / 전체 픽셀 수
- 평균 정확도 (Mean Accuracy)
    - 픽셀 정확도를 계산하는 방법
    - 평균 정확도 = (A 적중 픽셀수/전체 픽셀수 + B 적중 픽셀수 / 전체 픽셀수 + ...) / N
- MeanIU
    - 틀린 픽셀에 대한 정보가 반영됨
    - 틀린 픽셀수가 많아질 수록 값이 낮아짐
    - IU(Intersection over Union) : 특정 부분에서의 실제 픽셀과 예측 픽셀간의 합집합 영역 대비 교집합 영역의 비율
    - **MeanIU = (A IU + B IU + ...) / N** : 각각 구한 IU의 평균값. 
    - Frequency Weighted IU : 클래스별로 픽셀 수가 다를 경우, 픽셀 수가 많은 클래스에 더 비중을 주고 싶을 때 사용

### 딥러닝 시스템 구조
만들고 싶은 시스템을 **목표 시스템**이라고 했을 때 **학습 segment**와 **판정 segment**으로 나누어진다.
#### 학습 Segment
- 학습용 센싱 Element : 학습 데이터 습득 > 센싱 데이터
- 데이터셋 생성 Element : 센싱 데이터에서 학습에 적합한 형태로 전처리를 수행
- 딥러닝 모델 학습 Element : 딥러닝 모델 학습
#### 판정 Segment
- 판정용 센싱 Element : 실무 환경에서 데이터 수집 > 센싱 데이터
- 딥러닝 모델 판정 Element : 이미 학습된 딥러닝 모델을 이용해 센싱 데이터를 판정

### 학습 모델 저장
모델은 크게 **모델 아키텍처**와 **모델 가중치**로 구성된다.
- 모델 아키텍처(Model Architecture) : 모델의 구성 정의. 모델이 어떤 층으로 쌓여있는가 
- 모델 가중치(Weight) : 훈련셋으로 학습하면서 갱신 된 가중치
#### 모델 저장
즉, 학습 된 모델을 저장한다는 말은 '모델 아키텍처'와 '모델 가중치'를 저장한다는 말이다. 케라스에서는 **save()** 함수를 이용해 한번에 저장할 수 있다.
```python
from keras.models import load_model

model.save('mnist_mlp_model.h5')
``` 
- 모델 아키텍처와 모델 가중치 따로 저장하는 법
    - 모델 아키텍처 : model.to_json() / model.to_yaml() 함수 사용 -> .json / .yaml
    - 모델 가중치 : model.save_weights() 함수에 파일 경로를 인자로 입력 -> .h5
    - 따로 저장한 경우에는 구성도 따로
    - 모델 아키텍처를 먼저 구성한 뒤 가중치를 불러와 모델에 셋팅한다.
    ```python
    from models import model_from_json
    json_string = model.to_json()         # 모델 아키텍처를 json 형식으로 저장
    model = model_from_json(json_string)  # json 파일에서 모델 아키텍처 재구성
    
    from models import model_from_yaml
    yaml_string = model.to_yaml()         # 모델 아키텍처를 yaml 형식으로 저장
    model = model_from_yaml(yaml_string)  # yaml 파일에서 모델 아키텍처 재구성
    ```

#### 모델 아키텍처 보기
케라스에서는 모델을 객체를 생성한 이후에 **model_to_dat()** 함수를 이용하면 모델 아키텍처를 블록 형태로 가시화 할 수 있다.
```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
``` 
#### 모델 불러오기
저장한 모델은 당연히 불러와서 사용할 수 있다.
1. 모델을 불러오는 함수를 사용하기 전, 저장한 모델 파일로부터 모델을 재형성.
2. 실제 데이터로 모델을 사용. 주로 **predict()** 함수를 사용하지만, Sequentail 기반의 분류 모델의 경우에는 **predict_classes()** 함수를 제공한다. **predict_classes()** 는 가장 확률이 높은 클래스 인덱스를 알려준다.
```python
# 1. 실무에 사용할 데이터 준비

# 2. 모델 불러오기 
from keras.models import load_model
model = load_model('mnist_mlp_model.h5')

# 3. 모델 사용
```
#### 요약
- 케라스를 이용해 모델의 구성 및 가중치 정보 외에도 학습 설정 및 상태를 저장할 수 있음.
- 즉, 모델을 불러온 후 재학습을 시킬 수 있다. 이를 통해 신규 데이터셋이 계속 발생하는 경우, 빈번한 재학습 및 평가를 효과적으로 할 수 있다.
- 일반적인 딥러닝 시스템에서는 학습 처리 시간을 단축시키기 위해 **GPU나 클러스터 장비**에서 학습이 이뤄지고, - 판정 과정은 학습된 모델 결과 파일을 이용해 **일반 PC 및 모바일, 임베디드 등**에서 이루어짐.
- 딥러닝 모델에 대한 연구도 중요하지만, 실무에 적용하기 위해서는 목표 시스템에 대한 설계도 중요! (도메인, 사용 목적 등에 따라 운영 시나리오 및 환경이 다양하기 때문)