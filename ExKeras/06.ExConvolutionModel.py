# 컨볼루션 신경망 모델을 통한 손그림 분류

# 0. 패키지 추가
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 0. 랜덤시드 고정
np.random.seed(3)

# 1. 훈련셋, 검증셋 준비
# .flow_from_directory(path, target_size, batch_size, class_mode)
# path : 이미지 경로
# target_size : 패치 이미지 크기. 원본 이미지 크기가 다르더라도 자동 조절
# batch_size : 배치 크기
# class_mode : 분류 방식
#   - categorical : 2D one-hot 보호화된 라벨 반환
#   - binary : 1D 이진 라벨 반환
#   - sparse : 1D 정수 라벨 반환
#   - None : 라벨 미반환
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    './data/handwriting_shape/train',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    './data/handwriting_shape/test',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical'
)

# 2. 모델 구성
# Convolution Layer : 필터 수 32개, 필터 크기 3*3, 입력 이미지 크기 24*24, 입력 이미지 채널 3개, 활성화 함수 'relu'
# Convolution Layer : 필터 수 64개, 필터 크기 3*3, 활성화 함수 'relu'
# Max Fooling Layer : 풀 크기 2*2
# Flatten Layer
# Dense Layer : 출력 뉴런 수 128개, 활성화 함수 'relu'
# Dense Layer : 출력 뉴런 수 3개, 활성화 함수 'softmax'
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(24, 24, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 모델 학습과정 설정
# loss : 현재 가중치 세트를 평가하는 데 사용할 손실 함수. 다중 클래스 문제이므로 'categorical_crossentropy'
# optimizer : 최적 가중치 검색하는데 사용되는 최적화 알고리즘. 효율적 경사 하강법 알고리즘인 'adam'
# metrics : 평가 척도. 분류 문제에서는 일반적으로 'accuracy'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto') # 조기종료 함수

# 4. 모델 학습
# 케라스에서 모델을 학습시킬 때 주로 fit() 함수를 사용하지만, generator로 생성된 배치로 학습시킬 경우에는 fit_generator()함수를 사용
# .fit_generator(training_generator, steps_per_epoch, epochs, validation_data, validation_steps)
# training_generator : 훈련데이터셋을 제공할 제너레이터
# steps_per_epoch : 한 epoch에 사용할 스텝 수 (= 훈련 샘플 수/배치 사이즈)
# epochs : 전체 훈련 데이터셋의 학습 반복 횟수
# validation_data : 검증데이터셋을 제공할 제너레이터
# validation_steps : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수 (= 검증 샘플 수/배치 사이즈)
model.fit_generator(
        train_generator,
        steps_per_epoch=15,
        epochs=100,
        validation_data=test_generator,
        validation_steps=5
)

# 5. 모델 평가
# 케라스에서 모델을 평가할 때 주로 evaluate() 함수를 사용하지만, generator로 생성된 배치로 학습시킨 모델은 evaluate_generator() 함수를 사용
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s : %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 6. 모델 사용
# 제너레이터에서 제공되는 샘플을 입력할 때는 predict_generator() 함수를 사용.
# 예측 결과는 클래스별 확률 벡터로 출력된다.
# 클래스에 해당하는 열을 알기 위해서는 제너레이터의 'class_indices'를 출력하면 된다.
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)


# 이 모델의 경우, 매우 제한적인 학습 데이터셋만을 가지고 학습하기 때문에, 그리고 검증 데이터셋도 매우 유사하기 때문에 100%라는 정확도가 나온다.
# 하지만, 실제 현실에서는 이렇게 단순하지 않기 때문에 학습하는 데 있어 매우 많은 상황이 있을 수 있으며, 예외적인 경우도 많이 존재할 것이다.