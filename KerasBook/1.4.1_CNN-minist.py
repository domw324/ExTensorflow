import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# 1. 데이터 셋 준비
# 1-1. mnist 데이터 세트 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-2. 레이블 개수 계산
num_labels = len(np.unique(y_train))

# 1-3. one-hot vector 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 1-4. 크기 재조정 & 정규화
image_size = x_train.shape[1] # 입력 이미지 차원
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_test = x_test.astype('float32') / 255

# 2. 모델
# 2-1. 신경망 매개변수 정의 (이미지는 그대로 (정사각형 회색조) 처리됨)
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.2

# 2-2. 모델 생성 : CNN-ReLu-MaxPooling 스택
model = Sequential()
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(Flatten())
# 2-3. 드롭아웃 추가 - 정규화
model.add(Dropout(dropout))
# 2-4. 출력 계층 추가 - 10개 요소 one-hot vector
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file='cnn-mnist.png', show_shapes=True)

# 3. 모델학습방법 정의 : one-hot vector 위한 손실함수, adam 최적화, 분류 작업의 지표로 정확도 사용하는 것이 좋음
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 4. 신경망 훈련
model.fit(x_train, y_train, epochs=10, batch_size=batch_size)

# 5. 신경망 검증
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy : %.1f%%" % (100.0 * acc))