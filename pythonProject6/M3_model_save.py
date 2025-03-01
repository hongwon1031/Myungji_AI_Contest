import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 상수 정의
DATA_DIR_BASE = 'D:/ansys/new/'  # 데이터 디렉토리 기본 경로
TEMPERATURES = [25, 35, 45, 55, 65, 75, 85, 95, 105, 115,125,135,145,155,165,175,185,195,215,225]  # 온도 리스트
TOTAL_SIZE = 180  # 각 온도에 대한 총 데이터 포인트 수
PREDICTION_RANGE = 300  # 예측할 시간의 범위 (초)
FUTURE_STEPS = PREDICTION_RANGE - TOTAL_SIZE  # 예측할 미래 스텝 수

# JSON 데이터를 로드하는 함수
def load_temperature_data(data_dir_base, temperatures, total_size):
    all_data = []
    all_labels = []
    all_times = []
    for temp in temperatures:
        data_dir = f"{data_dir_base}/{temp}/json"
        for i in range(1, total_size + 1):
            file_path = os.path.join(data_dir, f'MINMAX_{i}.json')
            with open(file_path, 'r') as f:
                data = json.load(f)
                max_value = data['MAXIMUM VALUES']['VALUE']
                all_data.append(max_value)
                all_labels.append(temp)
                all_times.append(i)  # 시간 값을 추가
    return np.array(all_data).reshape(-1, 1), np.array(all_labels).reshape(-1, 1), np.array(all_times).reshape(-1, 1)

# 데이터 로드
data, temperatures, times = load_temperature_data(DATA_DIR_BASE, TEMPERATURES, TOTAL_SIZE)

# 데이터 정규화
scaler_data = MinMaxScaler(feature_range=(0, 1))
scaler_temps = MinMaxScaler(feature_range=(0, 1))
scaler_times = MinMaxScaler(feature_range=(0, 1))

data_scaled = scaler_data.fit_transform(data)
temps_scaled = scaler_temps.fit_transform(temperatures)
times_scaled = scaler_times.fit_transform(times)


# 데이터와 레이블 통합 (온도와 시간만 feature로 사용)
combined_data = np.hstack([temps_scaled, times_scaled])

# 데이터 샘플 생성 함수
def make_samples(features, labels, window_size):
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])  # (온도, 시간)
        y.append(labels[i])  # VALUE
    return np.array(X), np.array(y)

# 데이터 샘플 생성
WINDOW_SIZE = 125
X, y = make_samples(combined_data, data_scaled, WINDOW_SIZE)

# 학습 및 테스트 데이터 분할
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM 모델 구축
model = Sequential()
model.add(LSTM(32, input_shape=(WINDOW_SIZE, 2), activation='tanh', return_sequences=False))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_train, y_train, epochs=300, batch_size=16)

# 테스트 데이터에 대한 예측 수행
predictions_test = model.predict(X_test)

# 예측된 결과와 실제 결과를 원래 스케일로 변환
predictions_test_original = scaler_data.inverse_transform(predictions_test)
y_test_original = scaler_data.inverse_transform(y_test)

# 성능 평가 지표 계산
mse = mean_squared_error(y_test_original, predictions_test_original)
mae = mean_absolute_error(y_test_original, predictions_test_original)
r2 = r2_score(y_test_original, predictions_test_original)

print(f'MSE: {mse:.4e}')
print(f'MAE: {mae:.4e}')
print(f'R²: {r2:.4f}')

model_save_path = 'D:/ansys/new/Max_model/Max_lstm_model.h5'  # 모델을 저장할 경로
model.save(model_save_path)
print(f'Model saved to {model_save_path}')

#%% 예측
ONE_TEMPERATURES = 75
# 55도 데이터만 로드하여 예측을 준비
data_55, temperatures_55, times_55 = load_temperature_data(DATA_DIR_BASE, [ONE_TEMPERATURES], TOTAL_SIZE)
data_55_scaled = scaler_data.transform(data_55)
temps_55_scaled = scaler_temps.transform(temperatures_55)
times_55_scaled = scaler_times.transform(times_55)

combined_55 = np.hstack([temps_55_scaled, times_55_scaled])
X_55, _ = make_samples(combined_55, data_55_scaled, WINDOW_SIZE)

# 180초부터 300초까지의 예측
if len(X_55) >= 180-WINDOW_SIZE:  # 180 - WINDOW_SIZE
    future_predictions = []


    for step in range(FUTURE_STEPS):
        prediction = model.predict(X_55[-1].reshape(1, WINDOW_SIZE, 2))
        future_predictions.append(prediction[0][0])


        next_time = times_55_scaled[-1][0] + (step + 1) / TOTAL_SIZE
        new_input = np.array([[temps_55_scaled[-1][0], next_time]]).reshape(1, 2)
        X_55 = np.append(X_55, [np.vstack((X_55[-1][1:], new_input))], axis=0)

    future_predictions_original = scaler_data.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    selected_indices = np.where(temperatures == ONE_TEMPERATURES)[0]
    data_selected = data[selected_indices]
    time_range_55 = np.arange(181, 301)
    # 결과 시각화
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(1, len(data_selected) + 1), data_selected, label=f'Actual Data for {ONE_TEMPERATURES} degrees', linestyle='-')
    plt.plot(time_range_55, future_predictions_original, label=f'Predicted Data for {ONE_TEMPERATURES} degrees',
             linestyle='--')
    plt.title(f'Complete Temperature Data Prediction including Actual Data for {ONE_TEMPERATURES} degrees')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
else:
    print("Not enough data to predict 180 to 300 seconds.")
#%% 출력
min_value_index = np.argmin(np.abs(future_predictions_original))
min_value = future_predictions_original[min_value_index][0]
print("%d도의 열을 180초 가했을 때 최대 변위값은 %e이며" % (ONE_TEMPERATURES, data_selected[-1]))
if abs(data_selected[-1]) > abs(min_value):
    print('%d초 열을 더 가했을 시 %e로 변위값을 줄일 수 있습니다' % (min_value_index + 1 , min_value))
else:
    print('열을 더 가할 필요 없습니다')
    print("최소 절대값:", min_value)
    print("181 ~ 300초중 가장 낮은 값 인덱스:", min_value_index)






