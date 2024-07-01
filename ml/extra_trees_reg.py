import itertools
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import os


# Функция для создания дополнительных признаков
def create_features(df, target):
    df['month'] = df['reportts'].dt.month
    df['day'] = df['reportts'].dt.day
    df['day_of_week'] = df['reportts'].dt.dayofweek
    df['week_of_year'] = df['reportts'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['lag_1'] = target.shift(1)
    df['lag_2'] = target.shift(2)
    df['lag_3'] = target.shift(3)
    df['lag_4'] = target.shift(4)
    df['lag_5'] = target.shift(5)
    df['rolling_mean_3'] = target.rolling(window=3).mean()
    df['rolling_mean_7'] = target.rolling(window=7).mean()
    df['rolling_mean_14'] = target.rolling(window=14).mean()
    df['rolling_std_3'] = target.rolling(window=3).std()
    df['rolling_std_7'] = target.rolling(window=7).std()
    df['rolling_std_14'] = target.rolling(window=14).std()
    return df


# Функция для обучения модели и предсказания значений
def train_model(X, y):
    assert len(X) == len(y)

    # Создание дополнительных признаков
    X = create_features(X, y)

    # Предобработка данных
    X = X.fillna(0).drop(columns=['reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'])
    y = y.fillna(0)

    # Нормализация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Разделение данных на обучающую и тестовую выборки по времени
    split_index = int(len(X) * 0.75)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Создание модели
    et = ExtraTreesRegressor(random_state=42)

    # Гиперпараметрическая оптимизация с использованием кросс-валидации
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=et, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_et = grid_search.best_estimator_

    # Обучение модели на лучших параметрах
    best_et.fit(X_train, y_train)

    # Предсказание
    predictions = best_et.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    print(rmse, mae, len(predictions))
    return rmse, mae, best_et, y_val.index, predictions


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
        rmse, mae, model, indices, predictions = train_model(X, y)

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
    result_df.to_csv('predicted_data/extra_trees_reg.csv', index=False)


def create_etr_plots() -> go.Figure:
    csv_file_path = 'predicted_data/extra_trees_reg.csv'

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