import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Пути к загруженным файлам
X_train_path = './data/X_train.csv'
y_train_path = './data/y_train.csv'
X_test_path = './data/X_test.csv'

# Чтение данных
X_train = pd.read_csv(X_train_path, parse_dates=['reportts'])
y_train = pd.read_csv(y_train_path, parse_dates=['reportts'])
X_test = pd.read_csv(X_test_path, parse_dates=['reportts'])

# Проверка, что данные загружены корректно
print(X_train.head())
print(y_train.head())
print(X_test.head())

# Объединение датасетов
dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

# Убедитесь, что индекс поддерживается и добавьте частоту
dataset['reportts'] = pd.to_datetime(dataset['reportts'])
dataset = dataset.set_index('reportts')
dataset = dataset.asfreq('D')  # 'D' означает дневную частоту, можно изменить на 'H' для часовой или 'M' для месячной

# Обучение моделей для каждого из четырех графиков
fleet = ['VQ-BGU', 'VQ-BDU']
positions = [1, 2]
results = {}

for acnum in fleet:
    for pos in positions:
        key = f'{acnum}_pos_{pos}'
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']

        # Разбиение данных на обучающую и тестовую выборки
        train_size = int(len(y) * 0.75)
        train, test = y.iloc[:train_size], y.iloc[train_size:]

        # Обучение модели ARIMA
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()

        # Предсказание
        predictions = model_fit.forecast(steps=len(test))

        # Расчет метрик
        rmse = mean_squared_error(test, predictions, squared=False)  # использование squared=False для RMSE
        mae = mean_absolute_error(test, predictions)

        results[key] = (rmse, mae, test.index, predictions)

# Построение графиков
plt.figure(figsize=(14, 10))

for i, (label, group) in enumerate(results.items(), 1):
    rmse, mae, indices, predictions = group
    data_group = y_train[(y_train['acnum'] == label.split('_')[0]) & (y_train['pos'] == int(label.split('_')[2]))]

    # Отображение всех реальных значений
    plt.subplot(2, 2, i)
    plt.plot(data_group['egtm'].values, label=f'{label} Actual')

    # Выделение последних 25% реальных значений для совпадения с предсказанными
    val_start_idx = len(data_group) - len(predictions)
    plt.plot(range(val_start_idx, len(data_group)), predictions, label=f'{label} Predicted', color='orange')

    plt.title(f'{label} RMSE={rmse:.3f} MAE={mae:.3f}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

