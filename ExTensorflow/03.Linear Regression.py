# 선형회귀(Linear Regression) :
# 관찰된 연속형 변수들에 대해 두 변수 사이의 모형을 구한 뒤 적합도를 측정해내는 분석 방법
# 입력자료(독립변수) x 와 이에 대응하는 출력자료(종속변수) y 간의 관계를 정량화 하기 위한 작업
# 시간에 따라 변화하는 데이터나 어떤 영향, 가설적 실험, 인과 관계의 모델링 등의 통계적 예측에 이용될 수 있음

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def generate_dataset():
    x = np.linspace(0, 2, 100)
    y = 1.5 * x + np.random.randn(*x.shape) * 0.2 + 0.5
    return x, y


def linear_regression(x, y):
    w = tf.Variable(np.random.normal(), name='w')
    b = tf.Variable(np.random.normal(), name="b")
    y_pred = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_mean(tf.square(y_pred - y))

    return y_pred, loss


def run():
    x_batch, y_batch = generate_dataset()

    for i in range(x_batch.__sizeof__()):
        # print('x = {}, y = {}'.format(x_batch[i], y_batch[i]))
        x = x_batch[i]
        y = y_batch[i]

        y_pred, loss = linear_regression(x, y)

        # optimizer = tf.train.
        # train_op = optimizer.minimize(loss)

    plt.scatter(x_batch, y_batch)
    plt.title('Data')
    plt.show()

# 2.0 버전이라 진행이 잘 안된다.. keras를 사용하면 버전 상관없이 사용이 가능.. keras로 넘어가야겠다..

# if __name__ == "__main__":
run()
