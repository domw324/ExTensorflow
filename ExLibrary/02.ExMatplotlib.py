from matplotlib import pyplot as plt

# Case_01) Matplotlib는 수치 처리를 위해 일반적으로 Numpy 배열을 사용한다.
plt.plot(["Seoul", "Daejeon", "Busan"], [95, 137, 116])  # 수치 표시 : 꺽은선 (Default)
# plt.plot(["Seoul", "Daejeon", "Busan"], [95, 137, 116], 'ro') # 수치 표시 : 빨간 점
plt.xlabel('City')
plt.ylabel('Response')
plt.title('DeepLearning Result')
# plt.show() # 주석 해제하면 표시된다. 편의상 주석처리

# Case_02) Matplotlib는 여러개의 차트를 한번에 그리기 위해서 subplot() 사용
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(1, figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
# plt.show() # 주석 해제하면 표시된다. 편의상 주석처리

import numpy as np
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

# Case_03) 각종 연산을 그래프로 나타낼 수 있음
np.random.seed(19680801) # Fixing random state for reproducibility

# make up some data in the interval ]0, 1[
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

plt.figure(1) # plot with various axes scales

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)  # 그래프 눈금

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

plt.suptitle('Logit example')
plt.show()
