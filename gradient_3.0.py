import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Пути к загруженным файлам
X_train_path = './data/X_train.csv'
y_train_path = './data/y_train.csv'
X_test_path = './data/X_test.csv'

# Функция для обучения модели и предсказания значений
def train_model(X, y):
    # Проверка, что длины X и y совпадают
    assert len(X) == len(y)
    # Заполнение пропусков нулями и удаление ненужных колонок
    X = X.fillna(0).drop(columns=['reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'])
    # Определение индекса для разбиения данных на обучающую и валидационную выборки (75% обучающая, 25% валидационная)
    train_i = int(len(X) * 75 / 100)
    # Разбиение данных
    X_train, y_train = X.iloc[:train_i], y.iloc[:train_i]
    X_val, y_test = X.iloc[train_i:], y.iloc[train_i:]

    # Создание и обучение модели градиентного бустинга
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Предсказание значений на валидационной выборке
    predicted = model.predict(X_val)
    # Расчет метрик RMSE и MAE
    rmse = mean_squared_error(y_test, predicted, squared=False)
    mae = mean_absolute_error(y_test, predicted)

    # Возвращение метрик, модели, индексов валидационной выборки и предсказанных значений
    return rmse, mae, model, X_val.index, predicted

# Чтение данных из CSV файлов
X_train = pd.read_csv(X_train_path, parse_dates=['reportts'])
y_train = pd.read_csv(y_train_path, parse_dates=['reportts'])
X_test = pd.read_csv(X_test_path, parse_dates=['reportts'])

# Проверка, что данные загружены корректно (вывод первых 5 строк каждого датафрейма)
print(X_train.head())
print(y_train.head())
print(X_test.head())

# Объединение обучающего набора признаков и целевых значений по общим колонкам и удаление строк с пропусками в 'egtm'
dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

# Обучение моделей для каждого из четырех графиков (по комбинациям самолета и позиции)
fleet = ['VQ-BGU', 'VQ-BDU']
positions = [1, 2]
results = {}

for acnum in fleet:
    for pos in positions:
        # Создание ключа для идентификации комбинации самолета и позиции
        key = f'{acnum}_pos_{pos}'
        # Отбор данных для данной комбинации
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
        # Обучение модели и получение результатов
        rmse, mae, model, indices, predictions = train_model(X, y)
        # Сохранение результатов в словарь
        results[key] = (rmse, mae, indices, predictions)

# Чтение данных для графиков
df = pd.read_csv(y_train_path)
# Названия колонок для целевых значений и идентификации комбинаций
egtm_column_name = df.columns[3]
acnum_column_name = 'acnum'
pos_column_name = df.columns[2]

# Построение графиков
plt.figure(figsize=(14, 10))

for i, (label, group) in enumerate(results.items(), 1):
    rmse, mae, indices, predictions = group
    # Отбор данных для данной комбинации самолета и позиции
    data_group = df[(df[acnum_column_name] == label.split('_')[0]) & (df[pos_column_name] == int(label.split('_')[2]))]

    # Отображение всех реальных значений
    plt.subplot(2, 2, i)
    plt.plot(data_group[egtm_column_name].values, label=f'{label} Actual')

    # Выделение последних 25% реальных значений для совпадения с предсказанными
    val_start_idx = len(data_group) - len(predictions)
    plt.plot(range(val_start_idx, len(data_group)), predictions, label=f'{label} Predicted', color='orange')

    # Настройка заголовка графика и легенды
    plt.title(f'{label} RMSE={rmse:.3f} MAE={mae:.3f}')
    plt.legend()
    plt.grid(True)

# Настройка макета и отображение графиков
plt.tight_layout()
plt.show()
