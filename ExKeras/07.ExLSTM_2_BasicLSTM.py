# 동요 "나비야" 악보를 학습
# 음계를 입력하면 1. 다음 음계 예측 & 2. 1을 바탕으로 곡 전체 예측
# 모델 : LSTM 모델 / 128 메모리셀 LSTM 레이어 1개 + Dense 레이어, (샘플 50개, 타임스텝 4개, 속성 1개)입력, stateful=False

# 패키지
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import numpy as np

# 랜덤 시드 고정
np.random.seed(5)

# 손실 이력 클래스 정의
class CLossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 데이터셋 생성 함수 정의
def Seq2Dataset(seq, window_size):
    dataset = []
    for i in range(len(seq) - window_size):
        subset = seq[i:(i + window_size + 1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)

# 1. 데이터 준비
# 1-1. 코드 사전 정의
code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}
idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11: 'g8', 12:'a8', 13:'b8'}

# 1-2. 시퀀스 데이터 정의
seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# 2. 데이터셋 생성
dataset = Seq2Dataset(seq, window_size=4)

print(dataset.shape)
print(dataset)

# 2-1. 입력(x)과 출력(y) 변수로 분리
x_train = dataset[:, 0:4]
y_train = dataset[:, 4]

# 2-2. 입력값 정규화
max_idx_value = 13 # 정의한 코드가 0에서 13까지의 값을 가지기 때문에 max index = 13. [0/13 = 0, 1/13 = 0.077, ..., 13/13 = 1] 이므로 0~1까지 값으로 구분 가능
x_train = x_train / float(max_idx_value)
x_train = np.reshape(x_train, (50, 4, 1)) # (샘플 수, 타임스텝, 속성) 형태로 변환

# 2-3. 라벨값 one-hot 인코딩
y_train = np_utils.to_categorical(y_train) # y_train 값을 카테고리화. 표현하고 싶은 인덱스에만 1, 나머지는 0 처리
one_hot_vec_size = y_train.shape[1]
print("one hot encoding vector size is ", one_hot_vec_size)

# 3. 모델 구성
model = Sequential()
model.add(LSTM(128, input_shape=(4, 1))) # 메모리셀=128, (타임스텝=4, 속성=1)
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습
history = CLossHistory() # 손실 이력 객체 생성
model.fit(x_train, y_train, epochs=2000, batch_size=1, verbose=2, callbacks=[history]) # stateless, batch_size=1 : 매 샘플마다 상태가 초기화 된 채로 학습

# 6. 학습 과정 표현
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가
scores = model.evaluate(x_train, y_train)
print("%s : %.2f%%" %(model.metrics_names[1], scores[1]*100))

# - - - - 학습 완료 - - - -

# 8. 모델 사용하기
pred_count = 50 # 최대 예측 개수 정의

# 8-1. 한 스텝 예측
seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(x_train)
for i in range(pred_count):
    idx = np.argmax(pred_out[i])    # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx])   # 인덱스 값을 코드로 변환, 저장. seq_out은 최종 악보
print("one step prediction : ", seq_out)

# 8-2. 곡 전체 예측
seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in
seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in] # 코드를 인덱스값으로 변환
for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, 4, 1)) # (샘플 수, 타임스텝 수, 속성 수)
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)
print("full song prediction : ", seq_out)