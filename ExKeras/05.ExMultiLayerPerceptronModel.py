# 다층 퍼셉트론 모델 실습
# 모델을 만들 때 Basic Work-flow
# 문제 정의 -> 데이터 준비 -> 데이터셋 생성 -> 모델 구성 -> 모델 학습과정 설정 -> 모델 학습 -> 모델 평가 (-> 시각화)

# Load Packages
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# fix a Random Seed
np.random.seed(5)

# 1. Load a Data file
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",") # data 폴더의 csv 파일을 읽는다. 구분자=','

# 2. Create a DataSet
x_train = dataset[:700, 0:8]
y_train = dataset[:700, 8]  # the Number of DataSet for training = 700
x_test = dataset[700:, 0:8]
y_test = dataset[700:, 8]   # the Number of DataSet for test = 68 (after index 700 in the file)

# 3. Configuration a Model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))    # 입력층. Dense Layer 사용. 입력층 활성화 함수 'relu' 사용
model.add(Dense(8, activation='relu'))                  # 은닉층. Dense Layer 사용. 은닉층 활성화 함수 'relu' 사용. (입력 데이터 개수 == 직전층 출력 수)
model.add(Dense(1, activation='sigmoid'))               # 출력층. Dense Layer 사용. 이진 클래스 문제의 출력층이므로 'sigmoid' 활성화 함수 사용

# 4. Set up model training options
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# loss : 현재 가중치 세트를 평가하는 데 사용한 손실 함수 / 이진 클래스 문제이므로 'binary_crossentropy'
# optimizer : 최적 가중치 검색에 사용되는 최적화 알고리즘 / 효율적인 경사 하강법 알고리즘 중 하나인 'adam' 사용
# metrics : 평가 척도 / 분류 문제에서는 일반적으로 'accuracy'로 지정

# 5. Training the Model
model.fit(x_train, y_train, epochs=1500, batch_size=64)
# epochs=1500 : 1500회 반복학습
# batch_size=64 : 배치 크기(가중치 업데이트 단위) 64개

# 6. Evaluate the model
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 참고 할 전체 소스
#
# # 0. 사용할 패키지 불러오기
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
#
# # 랜덤시드 고정시키기
# np.random.seed(5)
#
# # 1. 데이터 준비하기
# dataset = np.loadtxt("./warehouse/pima-indians-diabetes.data", delimiter=",")
#
# # 2. 데이터셋 생성하기
# x_train = dataset[:700,0:8]
# y_train = dataset[:700,8]
# x_test = dataset[700:,0:8]
# y_test = dataset[700:,8]
#
# # 3. 모델 구성하기
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# # 4. 모델 학습과정 설정하기
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # 5. 모델 학습시키기
# model.fit(x_train, y_train, epochs=1500, batch_size=64)
#
# # 6. 모델 평가하기
# scores = model.evaluate(x_test, y_test)
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))