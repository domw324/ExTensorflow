import pandas as pd
import numpy as np

dates = pd.date_range('20190101', periods=6)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

print(df)
print(df.head()) # 데이터 앞에서부터. 5개 표시 (Default. 개수는 파라메터로 지정 가능)
print(df.tail(3)) # 데이터 뒤에서부터. 3개 표시
print(df.index)
print(df.columns)
print(df.values)
print(df.describe()) # 간단한 통계정보
print(df.T) # 전치행렬
print(df.sort_index(axis=1, ascending=False)) # 정렬. 컬럼기준(axis=1). 오름차순(ascending=True)/내림차순(ascending=False)
print(df.sort_values(by='B')) # B열 기준 오른차순 정렬
print(df['A']) # A 컬럼만 선택
print(df[0:3]) # 0에서 3행까지 선택
print(df.loc[dates[0]]) # 첫번째 인덱스에 해당하는 모든 값
print(df.iloc[3:5, 0:2]) # 3~4행, 0~1열 값
print(df[df.A > 0]) # A열의 값이 양수인 데이터
print(df.mean()) # 평균
print(df.apply(np.cumsum))


print('- - - 적용 - - -')
import matplotlib.pyplot as plt

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9

print(ts.head())

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()


print('- - - 적용 심화 - - -')

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
df.plot()
plt.show()