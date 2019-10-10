import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# Cosine 함수를 0부터 10pi까지 20개 만든다.
x = np.linspace(0, 10 * np.pi, 20)
y = np.cos(x)

# interoperate 함수로 보간법을 적용하여 linear(선형보정) quadratic(부드러운 보정) 두가지 방법으로 만든다
fl = interp1d(x, y, kind='linear')
fq = interp1d(x, y, kind='quadratic')

xint = np.linspace(x.min(), x.max(), 1000)
yintl = fl(xint)
yintq = fq(xint)

# Plot the data and the interpolation
plt.plot(x, y, 'bo')  # input value 청색점으로 표현
plt.plot(xint, yintl, color='green', linewidth=2) # linear 값 녹색 직선으로 표현
plt.plot(xint, yintq, color='red', linewidth=2) # quadratic 값 적색 곡선으로 표현
plt.legend(['Input', 'Linear', 'Quadratic'])
plt.ylim(-2, 2)

plt.title('Interoperate')
plt.show()