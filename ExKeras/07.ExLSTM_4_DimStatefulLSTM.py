# 동요 "나비야" 악보를 학습
# 음계를 입력하면 1. 다음 음계 예측 & 2. 1을 바탕으로 곡 전체 예측
# 모델 : LSTM 모델 / 128 메모리셀 LSTM 레이어 1개 + Dense 레이어, (샘플 50개, 타임스텝 4개, 속성 2개)입력, stateful=True

# 패키지
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils

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
    dataset_X = []
    dataset_Y = []

    for i in range(len(seq) - window_size):
        subset = seq[i:(i + window_size + 1)]

        for si in range(len(subset)-1):
            features = Code2Features(subset[si])
            dataset_X.append(features)

        dataset_Y.append([code2idx[subset[window_size]]])   # []로 감싸 2차 리스트형태로 넣어준다. (dataset_X 와 형식 동일하게)

    return np.array(dataset_X), np.array(dataset_Y)

# 속성 변환 함수
def Code2Features(code):
    features = []
    features.append(code2scale[code[0]] / float(max_scale_value))
    features.append(code2length[code[1]])
    return features

# 1. 데이터 준비
# 1-1. 코드 사전 정의
code2scale = {'c':0, 'd':1, 'e':2, 'f':3, 'g':4, 'a':5, 'b':6}
code2length = {'4':0, '8':1}

code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}
idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
            7:'c8', 8:'d8', 9:'e8', 10:'f8', 11: 'g8', 12:'a8', 13:'b8'}

max_scale_value = 6.0

# 1-2. 시퀀스 데이터 정의
seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# 2. 데이터셋 생성
x_train, y_train = Seq2Dataset(seq, window_size=4)

# 2-1. 입력값 (샘플 수, 타임스텝, 특성 수) 형태 변환
x_train = np.reshape(x_train, (50, 4, 2)) # 2종류의 특성을 가지는 입력 값

# 2-2. 라벨값 one-hot 인코딩
y_train = np_utils.to_categorical(y_train)
one_hot_vec_size = y_train.shape[1]
print("one hot encoding vector size is ", one_hot_vec_size)

# 3. 모델 구성
model = Sequential()
model.add(LSTM(128, batch_input_shape=(1, 4, 2), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습
history = CLossHistory() # 손실 이력 객체 설정

num_epochs = 2000
for epoch_idx in range(num_epochs):
    print("epochs : " + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=False, callbacks=[history]) # 50 is X.Shape[0]
    model.reset_states()

# 6. 학습 과정 표현
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.xlabel('epoch')
plt.xlabel('loss')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가
scores = model.evaluate(x_train, y_train, batch_size=1)
print("%s : %.2f%%" %(model.metrics_names[1], scores[1]*100))
model.reset_states()

# - - - - 학습 완료 - - - -

# 8. 모델 사용하기
pred_count = 50 # 최대 예측 개수 정의

# 8-1. 한 스텝 예측
seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(x_train, batch_size=1)
for i in range(pred_count):
    idx = np.argmax(pred_out[i])    # on-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx])   # 인덱스 값을 코드로 변환 저장. seq_out : 최종 악보

print("one step prediction : ", seq_out)
model.reset_states()

# 8-2. 곡 전체 예측
seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in
seq_in_features = []

for si in seq_in:
    features = Code2Features(si)
    seq_in_features.append(features)

for i in range(pred_count):
    sample_in = np.array(seq_in_features)
    sample_in = np.reshape(sample_in, (1, 4, 2)) # (샘플 수, 타임스텝, 속성 수)
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])

    features = Code2Features(idx2code[idx])
    seq_in_features.append(features)
    seq_in_features.pop(0)

print("full song prediction : ", seq_out)
model.reset_states()