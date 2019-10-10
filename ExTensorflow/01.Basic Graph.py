# Graphs
import tensorflow as tf

# 1-1. 기본 그래프
print('** 1-1. 기본 그래프 **')
a = 2
b = 3


@tf.function
def Add(x, y):
    return x + y


out = Add(a, b)  # tensorflow 2.o version 이라 Session이 필요없다.
print(out)

# 1-2. 조금 더 복잡한 그래프
print('** 1-2. 조금 더 복잡한 그래프 **')
a = 2
b = 3


@tf.function
# def Add(x, y):
#     return x + y
def Multiply(x, y):
    return x * y


def Power(x, n):
    return tf.pow(x, n, name="pow_op")


add_op = Add(a, b)
mul_op = Multiply(a, b)
pow_op = Power(add_op, mul_op)
useless_op = Multiply(a, add_op)

print(add_op)
print(mul_op)
print(pow_op)
print(useless_op)
