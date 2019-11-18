import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# 데이터셋 준비
# 1-1. MNIST 데이터 로딩
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train)
print(y_train)
print(x_test)
print(y_test)

# 1-2. 레이블 개수 계산
num_labels = len(np.unique(y_train))

# 1-3. 출력 데이터 one-hot vector 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 데이터셋 정규화
# 2-1. 이미지 차원 정의 (정사각형 가정)
image_size = x_train.shape[1]
input_size = image_size * image_size

# 2-2. 입력 값 크기 조정 & 정규화
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# 3. 모델 정의
# 3-1. 신경망 매개변수 정의
batch_size = 128
hidden_units = 256 # 256 units : 속도, 정확도 성능 탁월 / 128 : 신경망 빠르게 수렴, 정확도 낮음 / 512, 1024 : 테스트 정확도 크게 증가 X
dropout = 0.45 # 정규화 계층에 사용. 다음 계층으로 연결되는 유닛 중 임의로 제거할 유닛의 비율. 출력계층에는 사용하지 않음. (ex. dropout=0.45 : 다음 계층으로 열결되는 유닛 수 = 256*(1-0.45) = 140)

# 3-2. 모델 : 3개의 계층으로 이루어진 MLP 생성(각 계층 다음에는 ReLU와 드롭아웃 적용) (입력층, 은닉층)
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))

# 3-3. one-hot vector 출력 (출력층)
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file='mlt-mnist.png', show_shapes=True)

# 4. 모델 학습 방법 정의 (one-hot vector 손실 함수, adam 최적화 사용, 분류 작업의 지표로 정확도(accuracy) 사용)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 훈련
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

# 6. 모델 검증 (일반화가 제대로 됐는지 확인하기 위해 테스트 데이터세트로 검증)
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy : %.1f%%" %(100.0 * acc))