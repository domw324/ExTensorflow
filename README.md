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
