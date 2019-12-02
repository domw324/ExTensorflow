import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

# 1. 데이터세트 준비
# 1-1. mnist 데이터세트 로딩
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-2. 레이블 개수 계산
num_labels = len(np.unique(y_train))

# 1-3. 결과 데이터 ont-hot vector 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 1-4. 입력 이미지 형상 재조정 & 정규화
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_test = x_test.astype('float32') / 255

# 2. 모델
# 2-1. 신경망 매개변수 정의 : 이미지는 그대로 처리 (회색조, 정사각형 이미지)
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

# 2-2. 모델 정의 : 함수형 API 사용 CNN 계층 구축
inputs = Input(shape=input_shape)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(inputs)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)
y = Flatten()(y) # 밀집 계층 연결하기 전 이미지를 벡터로 변환
y = Dropout(dropout)(y) # 드롭아웃 정규화
outputs = Dense(num_labels, activation='softmax')(y)

model = Model(inputs=inputs, outputs=outputs) # 입력/출력을 제공해 모델 구축
model.summary() # 텍스트로 신경망 모델 요약

# 3. 모델 학습 방법 정의 : 분류 모델 손실 함수, Adam 최적화, 정확도
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 4. 모델 훈련 : 입력 이미지와 레이블로 모델 훈련
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=batch_size)

# 5. 모델 검증
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy : %.1f%%" % (100.0 * score[1]))


# ref)
# 기본적으로 MaxPooling2D는 pool_size=2를 사용하기때문에 인수가 제거됐다.
# 위 코드의 모든 계층은 텐서의 함수이다. 각 계층은 다음 계층의 입력이 될 출력으로 텐서를 생성.
# 모델 생성 위해서 inputs 텐서와 ouputs 텐서를 공급하거나, 텐서 리스트를 제공해 Model()을 호출하면 된다.
# 순차형 모델과 비슷하게, 같은 코드를 fit()과 evaluate()함수를 사용해 훈련&평가 할 수 있다. (Sequntial 클래스는 실제로 Model 클래스의 하위 클래스이다.)
# 모델을 훈련시키는 동안 검증 정확도가 어떻게 진화하는지 확인하기 위해 fit()함수에 validation_data 인수를 삽입함.