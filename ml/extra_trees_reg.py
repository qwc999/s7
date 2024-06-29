import itertools
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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
    grid_search = GridSearchCV(estimator=et, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_et = grid_search.best_estimator_
    
    # Обучение модели на лучших параметрах
    best_et.fit(X_train, y_train)
    
    # Предсказание
    predictions = best_et.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    
    return rmse, mae, best_et, y_val.index, predictions

# Основная функция для выполнения всех шагов и построения графиков
def create_etr_plots():
    # Чтение данных
    X_train = pd.read_csv('./data/X_train.csv', parse_dates=['reportts'])
    y_train = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
    X_test = pd.read_csv('./data/X_test.csv', parse_dates=['reportts'])

    dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

    # Обучение моделей для каждого из четырех графиков
    fleet = ['VQ-BGU', 'VQ-BDU']
    positions = [1, 2]
    results = {}

    for acnum, pos in itertools.product(fleet, positions):
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
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f'{key} RMSE={results[key][0]:.3f} MAE={results[key][1]:.3f}'
            for key in results
        ],
    )

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

        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        fig.add_trace(go.Scatter(x=dates, y=actual_values, mode='lines', name=f'{label} Actual', line=dict(color='blue')), row=row, col=col)
        fig.add_trace(go.Scatter(x=dates, y=predicted_values, mode='lines', name=f'{label} Predicted', line=dict(color='red', dash='dash')), row=row, col=col)

    fig.update_layout(title_text='EGTM Predictions', height=800, width=1200)
    return fig