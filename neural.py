import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Загрузка данных
X_train = pd.read_csv('./data/X_train.csv', parse_dates=['reportts'])
y_train = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
X_test = pd.read_csv('./data/X_test.csv', parse_dates=['reportts'])
dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])


# Определение функции для обучения нейросети
def train_model_nn(X, y):
    assert len(X) == len(y)

    # Предобработка данных
    X = X.fillna(0).drop(columns=['reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'])
    y = y.fillna(0)

    # Нормализация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    # Создание модели
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Компиляция модели
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Обучение модели
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    # Оценка модели
    predicted = model.predict(X_val)
    rmse = mean_squared_error(y_val, predicted, squared=False)
    mae = mean_absolute_error(y_val, predicted)

    for pred, true in zip(predicted, y_val):
        print(f'Predicted: {pred[0]:.3f}, True: {true:.3f}')

    return rmse, mae, model


# Запуск обучения для каждого самолета
fleet = ['VQ-BGU', 'VQ-BDU']
for acnum in fleet:
    X = dataset[dataset['acnum'] == acnum].drop(columns=['egtm'])
    y = dataset[dataset['acnum'] == acnum]['egtm']
    rmse, mae, model = train_model_nn(X, y)
    print(f'acnum={acnum} RMSE={rmse:.3f} MAE={mae:.3f}')
