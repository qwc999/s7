import os
import itertools
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def train_model_cv(X, y):
    # Проверка, что длины X и y совпадают
    assert len(X) == len(y)
    # Заполнение пропусков нулями и удаление ненужных колонок
    X = X.fillna(0).drop(columns=['reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'])

    # Создание и настройка модели байесовской регрессии
    model = BayesianRidge()

    # Определение кросс-валидации с 4 фолдами
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # Предсказание с использованием кросс-валидации
    predicted = cross_val_predict(model, X, y, cv=kf)
    n_last = int(0.25 * len(y))
    y_last = y[-n_last:]
    predicted_last = predicted[-n_last:]
    rmse = np.sqrt(mean_squared_error(y_last, predicted_last))
    mae = mean_absolute_error(y_last, predicted_last)
    print(rmse, mae)
    return rmse, mae, model, n_last, predicted_last


def save_predictions_to_csv():
    # Чтение данных
    X_train = pd.read_csv('./data/X_train.csv', parse_dates=['reportts'])
    y_train = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
    X_test = pd.read_csv('./data/X_test.csv', parse_dates=['reportts'])

    dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

    # Обучение моделей для каждого из четырех графиков
    fleet = ['VQ-BGU', 'VQ-BDU']
    positions = [1, 2]
    results = []

    for acnum, pos in itertools.product(fleet, positions):
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
        rmse, mae, model, indices, predictions = train_model_cv(X, y)

        # Добавление результатов в общий список
        results.append(pd.DataFrame({
            'reportts': dataset.loc[indices, 'reportts'],
            'acnum': acnum,
            'pos': pos,
            'egtm': predictions
        }))

    # Сохранение результатов в CSV файл
    result_df = pd.concat(results)
    os.makedirs('predicted_data', exist_ok=True)
    result_df.to_csv('predicted_data/bayesian_ridge.csv', index=False)


def create_bayesian_ridge_plots() -> go.Figure:
    csv_file_path = 'predicted_data/bayesian_ridge.csv'

    # Проверка на существование CSV файла
    if not os.path.exists(csv_file_path):
        save_predictions_to_csv()

    # Чтение данных для графиков
    df = pd.read_csv('data/y_train.csv')
    predicted_df = pd.read_csv(csv_file_path)
    egtm_column_name = df.columns[3]
    acnum_column_name = 'acnum'
    pos_column_name = df.columns[2]

    fleet = ['VQ-BGU', 'VQ-BDU']
    positions = [1, 2]
    graphs = []

    for i, (acnum, pos) in enumerate(itertools.product(fleet, positions), 1):
        data_group = df[(df[acnum_column_name] == acnum) & (df[pos_column_name] == pos)]

        # Найти соответствующий подмножество в результатах
        predicted_data = predicted_df[(predicted_df['acnum'] == acnum) & (predicted_df['pos'] == pos)]
        val_start_idx = len(data_group) - len(predicted_data)

        # Убедиться, что размеры совпадают
        graphs.append(
            [
                f'{acnum}_pos_{pos}',
                go.Scatter(
                    y=data_group[egtm_column_name].values,
                    name=f'{acnum}_pos_{pos}'
                ),
                go.Scatter(
                    x=list(range(val_start_idx, len(data_group))),
                    y=predicted_data['egtm'],
                    name=f'{acnum}_pos_{pos} Predicted',
                ),
            ]
        )

    fig = make_subplots(rows=2, cols=2, subplot_titles=[i[0] for i in graphs])

    for i in range(2):
        fig.add_trace(
            graphs[0][i + 1],
            row=1,
            col=1
        )
        fig.add_trace(
            graphs[1][i + 1],
            row=1,
            col=2
        )
        fig.add_trace(
            graphs[2][i + 1],
            row=2,
            col=1
        )
        fig.add_trace(
            graphs[3][i + 1],
            row=2,
            col=2
        )

    fig.update_layout(
        hovermode="x",
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig