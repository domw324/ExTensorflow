# Type
import tensorflow as tf

# 2-1. Constant : 상수, 값이 변경되지 않는 노드
print('** 2-1. Constant **')
a = tf.constant(2)
b = tf.constant(3)


@tf.function
def Add(x, y):
    return x + y


print(Add(a, b))

# 2-2. Variable : 변수, graph의 다양한 실행에 대한 값을 저장
print('** 2-2. Variable **')
# 변수 선언 방법
w = tf.Variable(2, name="scalar")
m = tf.Variable([[1, 2], [3, 4]], name="matrix")
W = tf.Variable(tf.zeros([784, 10])) # 0으로 초기화된 784*10 배열 선언

print(w)
print(m)
print(W)

# 변수 활용
weight = tf.Variable(tf.ones(shape=(2, 2)), name="W")
bias = tf.Variable(tf.zeros(shape=2), name="b")
print('weights = {}'.format(W))
print('biases = {}'.format(b))

@tf.function
def forward(x):
    return x * weight + bias


out_a = forward([1, 0])
print('Result = {}'.format(out_a))

# 2-3. 실습
print('** 2-3. 실습 **')

