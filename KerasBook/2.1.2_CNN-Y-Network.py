import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.layers.merge import concatenate
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model

# 1. 데이터 준비
# 1-1. MNIST 데이터세트 로딩
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-2. 레이블 개수 계산
num_labels = len(np.unique(y_train))

# 1-3. 출력값 on-hot vector 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 1-4. 입력 이미지 형상 재조정 & 정규화
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_test = x_test.astype('float32') / 255

# 2. 모델 정의
# 2-1. 매개변수 정의
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
dropout = 0.4
n_filters = 32

# 2-2. Y-Network의 왼쪽 가지 정의 (the Left Branch of Y-Network)
left_inputs = Input(shape=input_shape)
x = left_inputs
filters = n_filters
# Conv2D-Dropout-MaxPooling2D 3계층
# 계층 지날 때마다 필터 개수 두 배로 증가 (32-64-128)
for i in range(3):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2

# 2-3. Y-Network의 오른쪽 가지 정의 (the Right Branch of Y-Network)
right_inputs = Input(shape=input_shape)
y = right_inputs
filters = n_filters
# Conv2D-Dropout-MaxPooling2D 3계층
# 계층 지날 때마다 필터 개수 두 배로 증가 (32-64-128)
for i in range(3):
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(y)
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2

# 2-4. 왼쪽 가지와 오른쪽 가지의 출력을 병합
z = concatenate([x, y])

# 2-5. Dense 계층에 연결하기 전 특징 맵을 벡터로 변환
z = Flatten()(z)
z = Dropout(dropout)(z)
outputs = Dense(num_labels, activation='softmax')(z)

# 2-6. 함수형 API 모델 구축
model = Model([left_inputs, right_inputs], outputs)
plot_model(model, to_file='cnn-y-network.png', show_shapes=True) # 모델 확인 : 그래프 사용
model.summary() # 모델 확인 : 텍스트 요약

# 3. 모델 학습 방법 정의 : 분류 모델 손실 함수, Adam 최적화, 분류 정확도
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 4. 모델 훈련 (입력 이미지와 레이블로 모델 훈련)
model.fit([x_train, x_train],
          y_train,
          validation_data=([x_test, x_test], y_test),
          epochs=20,
          batch_size=batch_size)

# 5. 모델 평가 : 테스트 데이터세트에서 모델 정확도 측정
score = model.evaluate([x_test, x_test], y_test, batch_size=batch_size)
print("=nTest accuracy : %.1f%%" % (100 * score[1]))