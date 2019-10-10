import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense

training_date = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
target_date = np.array([[0], [1], [1], [0]], "float32")

# instance 생성 및 Layer 추가
model = Sequential()

model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 오차계산(loss), 학습방법(optimizer) 결정
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

# 데이터 입력, 학습
model.fit(training_date, target_date, epochs=1000, verbose=2)

print(model.predict(training_date))