import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# 1. 데이터셋 준비
# 1-1. mnist 데이터세트 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-2. 레이블 개수 계산
num_labels = len(np.unique(y_train))

# 1-3. 결과 데이터 one-hot vector 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 1-4. 입력 데이터 크기 재조정 & 정규화
image_size = x_train.shape[1]
x_train = np.reshape(x_train, (-1, image_size, image_size))
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, (-1, image_size, image_size))
x_test = x_test.astype('float32') / 255

# 2. 모델
# 2-1. 신경망 매개변수 정의
input_shape = (image_size, image_size)
batch_size = 238
units = 256
dropout = 0.2

# 2-2. 모델 정의 : 모델 - 256 유닛으로 구성된 RNN / 입력 - 28 시간 단계, 28 항목으로 구성된 벡터
model = Sequential()
model.add(SimpleRNN(units=units,
                    dropout=dropout,
                    input_shape=input_shape))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

# 3. 모델 학습 방법 정의 : one-hot vector 위한 솔실 함수, SGD 최적화, 분류 작업에 대한 지표로 정확도가 바람직
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 4. 모델 훈련
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

# 5. 모델 검증
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy : %.1f%%" % (100.0 * acc))