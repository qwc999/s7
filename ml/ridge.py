import itertools
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Функция для обучения модели и предсказания значений с использованием кросс-валидации
def train_model_cv(X, y):
    # Проверка, что длины X и y совпадают
    assert len(X) == len(y)
    # Заполнение пропусков нулями и удаление ненужных колонок
    X = X.fillna(0).drop(columns=['reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'])

    # Создание и настройка модели байесовской регрессии
    model = BayesianRidge()

    # Определение кросс-валидации с 5 фолдами
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Предсказание с использованием кросс-валидации
    predicted = cross_val_predict(model, X, y, cv=kf)

    # Расчет метрик RMSE и MAE
    rmse = mean_squared_error(y, predicted, squared=False)
    mae = mean_absolute_error(y, predicted)

    # Возвращение метрик и предсказанных значений
    return rmse, mae, predicted

def create_ridge_plots():
    # Пути к файлам
    X_train_path = './data/X_train.csv'
    y_train_path = './data/y_train.csv'
    X_test_path = './data/X_test.csv'

    # Чтение данных из CSV файлов
    X_train = pd.read_csv(X_train_path, parse_dates=['reportts'])
    y_train = pd.read_csv(y_train_path, parse_dates=['reportts'])
    X_test = pd.read_csv(X_test_path, parse_dates=['reportts'])

    # Объединение обучающего набора признаков и целевых значений по общим колонкам и удаление строк с пропусками в 'egtm'
    dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

    # Обучение моделей для каждого из четырех графиков (по комбинациям самолета и позиции)
    fleet = ['VQ-BGU', 'VQ-BDU']
    positions = [1, 2]
    figures = []

    for acnum, pos in itertools.product(fleet, positions):
        # Создание ключа для идентификации комбинации самолета и позиции
        key = f'{acnum}_pos_{pos}'
        # Отбор данных для данной комбинации
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
        # Обучение модели и получение результатов
        rmse, mae, predictions = train_model_cv(X, y)

        # Создание реальных значений для графика
        data_group = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]
        actual_values = data_group['egtm'].values

        # Определение индекса для отображения последних 25% реальных значений
        val_start_idx = int(0.75 * len(actual_values))

        # Создание фигуры Plotly для текущей комбинации
        fig = go.Figure()

        # Добавление реальных значений на график
        fig.add_trace(go.Scatter(x=list(range(len(actual_values))), y=actual_values,
                                mode='lines', name=f'{key} Actual'))

        # Добавление предсказанных значений на график (только последние 25%)
        fig.add_trace(go.Scatter(x=list(range(val_start_idx, len(predictions))),
                                y=predictions[val_start_idx:], mode='lines', name=f'{key} Predicted'))

        # Настройка заголовка графика
        fig.update_layout(title=f'{key} RMSE={rmse:.3f} MAE={mae:.3f}',
                        xaxis_title='Index', yaxis_title='EGTM Value')

        # Добавление фигуры в список
        figures.append((f'{key} RMSE={rmse:.3f} MAE={mae:.3f}', fig))

    # Распределение графиков по сетке 2x2
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[figures[0][0], figures[1][0], figures[2][0], figures[3][0]])

    # Задание расположения графиков
    fig.add_trace(figures[0][1].data[0], row=1, col=1)
    fig.add_trace(figures[0][1].data[1], row=1, col=1)
    fig.add_trace(figures[1][1].data[0], row=2, col=1)
    fig.add_trace(figures[1][1].data[1], row=2, col=1)
    fig.add_trace(figures[2][1].data[0], row=1, col=2)
    fig.add_trace(figures[2][1].data[1], row=1, col=2)
    fig.add_trace(figures[3][1].data[0], row=2, col=2)
    fig.add_trace(figures[3][1].data[1], row=2, col=2)

    # Обновление общего заголовка фигуры
    fig.update_layout(title_text='EGTM Predictions')

    return fig

