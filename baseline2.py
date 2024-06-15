"""
выводит графики с данными и предсказанными значениями
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Функция для обучения модели и предсказания значений
def train_model(X, y):
    assert len(X) == len(y)
    X = X.fillna(0).drop(columns=[
        'reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'
    ])
    train_i = int(len(X) * 75 / 100)
    X_train, y_train = X[0:train_i], y[0:train_i]
    X_val, y_test = X[train_i:], y[train_i:]

    model = Ridge(alpha=5)
    model.fit(X_train, y_train)

    predicted = model.predict(X_val)
    rmse = mean_squared_error(y_test, predicted, squared=False)
    mae = mean_absolute_error(y_test, predicted)

    return rmse, mae, model, X_val.index, predicted


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
        print(predictions)
        results[key] = (rmse, mae, indices, predictions)
        print(f'{key} RMSE={rmse:.3f} MAE={mae:.3f}')

# Чтение данных для графиков
df = pd.read_csv('data/y_train.csv')
egtm_column_name = df.columns[3]
acnum_column_name = 'acnum'
pos_column_name = df.columns[2]

# Построение графиков
plt.figure(figsize=(14, 10))

for i, (label, group) in enumerate(results.items(), 1):
    rmse, mae, indices, predictions = group
    data_group = df[(df[acnum_column_name] == label.split('_')[0]) & (df[pos_column_name] == int(label.split('_')[2]))]

    # Отображение всех реальных значений
    plt.subplot(2, 2, i)
    plt.plot(data_group[egtm_column_name].values, label=f'{label} Actual')

    # Выделение последних 25% реальных значений для совпадения с предсказанными
    val_start_idx = len(data_group) - len(predictions)
    plt.plot(range(val_start_idx, len(data_group)), predictions, label=f'{label} Predicted', color='orange')

    plt.title(f'{label} RMSE={rmse:.3f} MAE={mae:.3f}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
