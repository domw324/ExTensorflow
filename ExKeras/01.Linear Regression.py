import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from mpl_toolkits.mplot3d import Axes3D

raw_data = np.genfromtxt('x09.txt', skip_header=36)

x_data = np.array(raw_data[:, 2:4], dtype=np.float32)
y_data = np.array(raw_data[:, 4], dtype=np.float32)
y_data = y_data.reshape((25, 1))

rmsprop = RMSprop(lr=0.01)
model = Sequential()
model.add(Dense(1, input_shape=(2,)))
model.compile(loss='mse', optimizer=rmsprop)
model.summary()

hist = model.fit(x_data, y_data, epochs=1000)

print(hist.history.keys())

print("100Kg 40세 혈중지방함량치 = {}".format(model.predict(np.array([100, 40]).reshape(1, 2))))
print("60Kg 25세 혈중지방함량치 = {}".format(model.predict(np.array([60, 25]).reshape(1, 2))))

# === 입력 데이터를 보기위한 테스트 코드 ===
fig = plt.figure(figsize=(12, 12))
figure = fig.add_subplot(111, projection='3d')
figure.set_xlabel('Weight')
figure.set_ylabel('Age')
figure.set_zlabel('Blood fat')
figure.view_init(15, 15)

xs = np.array(raw_data[:, 2], dtype=np.float32)
ys = np.array(raw_data[:, 3], dtype=np.float32)
zs = np.array(raw_data[:, 4], dtype=np.float32)
figure.scatter(xs, ys, zs)

W, b = model.get_weights()
x = np.linspace(20, 100, 50).reshape(50, 1)
y = np.linspace(10, 70, 50).reshape(50, 1)
X = np.concatenate((x, y), axis=1)
Z = np.matmul(X, W) + b
figure.plot_wireframe(x, y, Z, color='red')

plt.show()