# ExConvolutionModel.py 파일에서 구현한 모델은 매우 한정적인 데이터셋과 검증셋으로 이루어졌다.
# 그래서 일반적이지 않은 데이터셋이 들어간 경우 매우 낮은 정확도를 나타낸다.
# 조금 더 다양한 학습과 검증을 하기 위해 "데이터 부풀리기"를 실습한다.

import numpy as np

# 랜덤시드 고정
np.random.seed(3)

# 데이터셋 불러오기
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.7,
    zoom_range=[0.9, 2.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
) # training DataSet 부풀리기 설정
train_generator = train_datagen.flow_from_directory(
    './data/handwriting_shape_hard/train',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    './data/handwriting_shape_hard/test',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical'
)

# 모델 구성
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(24, 24, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 학습방법 정의
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
from tensorflow import keras
# early_stopping = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto')]
# 학습 조기종료 함수에 patience가 있는 경우, (조기 종료 됐을 때)실제 가장 최적 모델은 멈춘 시기의 모델이 아니다.
# -5회 학습인 경우가 최적 모델인데, 이때의 상태를 항상 체크하여 저장하도록 해야한다.
early_stopping = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto'),
    keras.callbacks.ModelCheckpoint(filepath='best_model_convolution_hard.h5', monitor='val_loss', save_best_only=True)
]

model.fit_generator(
    train_generator,
    steps_per_epoch=15 * 100,
    epochs=200,
    validation_data=test_generator,
    validation_steps=5,
    callbacks=early_stopping
)

# 모델 평가
print("-- Evalute --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s : %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 모델 예측
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(output)