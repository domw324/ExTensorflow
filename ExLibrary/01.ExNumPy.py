import numpy as np

print("\n- - First - -\n")

a = np.array([20, 30, 40, 50])
b = np.arange(4)
print("배열 a", a)
print("arrange로 생성한 b", b)
c = a - b
print("c=a-b 의 결과값", c)
print("b**2 의 결과값", b ** 2)  # 행렬 각 요소 2제곱
print("a의 sine값 * 10 의 결과값", 10 * np.sin(a))
print("a가 35보다 작은지에 대한 비교 결과", a < 35)

AA = np.array([[1, 1], [0, 1]])
BB = np.array([[2, 0], [3, 4]])
print("AA * BB 의 결과값\n", AA * BB)  # 행렬 요소끼리 곱셈
print("AA @ BB 의 결과값\n", AA @ BB)  # 행렬 곱셈
print("AA.dot(BB) 의 결과값\n", AA.dot(BB))  # 행렬 곱셈

print("\n- - Second - -\n")

mtxA = np.arange(10) ** 3
print("1부터 10까지 3자승 한 배열 a =", mtxA)
print("mtxA 배열의 2번 인덱스 값 = ", mtxA[2])
print("mtxA 배열의 2:5 인덱스 값 = ", mtxA[2:5])

mtxA[0:6:2] = -1000  # from start to position 6, exclusive, set every 2nd element to -1000
print("mtxA 배열의 0번째부터 5번째까지 모든 두번째 값에 -1000을 대입한 값 = ", mtxA)
print("mtxA 배열을 뒤집은 값  = ", mtxA[::-1])  # Reversed a


# 각 행마다 10씩 증가하는 배열 생성
def CreateMatrix(x, y):
    return 10 * x + y

mtxB = np.fromfunction(CreateMatrix, (5, 4), dtype=int)
print("각 행마다 10씩 증가하는 배열 생성\n", mtxB)
print("생성된 배열의 2행 3열 값 = ", mtxB[2, 3])
print("생성된 배열 각행 2열 값 = ", mtxB[0:5, 1])
print("생성된 배열 각행 1행~2행 값 \n ", mtxB[1:3, :])


print("\n- - Third - -\n")

mtxC = np.floor(10*np.random.random((3, 4))) # 0 ~ 1 사이의 균등 분포로 표본을 추출
print("3행 4열 배열 생성\n", mtxC)
print("mtxC 배열의 shape = ", mtxC.shape)

print("mtxC.ravel 결과값\n", mtxC.ravel())  # returns the array, flattened. 1차원 배열로 변경
print("mtxC.reshape(6,2) 결과값\n", mtxC.reshape(6, 2))  # returns the array with a modified shape. 행과 열 모양 재설정
print("mtxC 전치행렬 mtxC.T\n", mtxC.T)  # returns the array, transposed. 전치행렬
print("mtxC 배열 모양 = ", mtxC.shape)
print("mtxC.T 배열 모양 = ", mtxC.T.shape)