# MNIST 데이터세트에 접속해서 그 중 임의로 25개의 샘플 이미지를 표시하고 훈련/테스트 데이터세트의 레이블 개수를 세는 방법을 보여준다.

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. 데이터 세트 로딩
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-1. 고유한 훈련 레이블 개수 세기
unique, counts = np.unique(y_train, return_counts=True)
print("Test labels : ", dict(zip(unique, counts)))

# 1-2. 고유한 테스트 레이블 개수 세기
unique, counts = np.unique(y_test, return_counts=True)
print("Test labels : ", dict(zip(unique, counts)))

# 2. 훈련 데이터세트에서 25개의 mnist 숫자 샘플 추출하기
indexs = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[indexs]
labels = y_train[indexs]

# 3. 25개의 mnist 숫자 그리기
plt.figure(figsize=(5, 5))
for i in range(len(indexs)):
    plt.subplot(5, 5, i+1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')

plt.show()
plt.savefig("mnist-samples.png")
plt.close('all')