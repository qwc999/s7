import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Функция для обучения модели и предсказания значений
def train_model(X, y):
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
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Предсказание
    predicted = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predicted))
    mae = mean_absolute_error(y_val, predicted)
    
    return rmse, mae, model, y_val.index, predicted

# Чтение данных
X_train = pd.read_csv('./data/X_train.csv', parse_dates=['reportts'])
y_train = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
X_test = pd.read_csv('./data/X_test.csv', parse_dates=['reportts'])

dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

# Обучение моделей для каждого из четырех графиков
fleet = ['VQ-BGU', 'VQ-BDU']
positions = [1, 2]
results = {}

for acnum in fleet:
    for pos in positions:
        key = f'{acnum}_pos_{pos}'
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
        rmse, mae, model, indices, predictions = train_model(X, y)
        results[key] = (rmse, mae, indices, predictions)
        print(f'{key} RMSE={rmse:.3f} MAE={mae:.3f}')

# Чтение данных для графиков
df = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
egtm_column_name = 'egtm'
acnum_column_name = 'acnum'
pos_column_name = 'pos'

# Построение графиков
plt.figure(figsize=(14, 10))

for i, (label, group) in enumerate(results.items(), 1):
    rmse, mae, indices, predictions = group
    acnum, _, pos = label.split('_')
    data_group = df[(df[acnum_column_name] == acnum) & (df[pos_column_name] == int(pos))]
    
    actual_values = data_group[egtm_column_name].values
    dates = data_group['reportts'].values
    
    # Создаем массив с предсказанными значениями такой же длины, как и actual_values
    predicted_values = np.full(len(actual_values), np.nan)
    
    # Соотносим индексы предсказанных значений с локальными индексами data_group
    for j, idx in enumerate(indices):
        if idx in data_group.index:
            predicted_values[data_group.index.get_loc(idx)] = predictions[j]

    plt.subplot(2, 2, i)
    plt.plot(dates, actual_values, label=f'{label} Actual', color='blue')
    plt.plot(dates, predicted_values, label=f'{label} Predicted', color='red', linestyle='--')

    plt.title(f'{label} RMSE={rmse:.3f} MAE={mae:.3f}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
