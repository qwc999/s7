import os
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Пути к файлам
X_train_path = './data/X_train.csv'
y_train_path = './data/y_train.csv'
X_test_path = './data/X_test.csv'
predicted_data_path = './predicted_data/BayesianRidge.csv'


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


# Функция для сохранения предсказанных данных в CSV файл
def save_predictions_to_csv(predictions, file_path):
    # Создание папки, если она не существует
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Создание DataFrame с предсказанными значениями
    predictions_df = pd.DataFrame(predictions, columns=['acnum_pos', 'actual', 'predicted'])

    # Сохранение DataFrame в CSV файл
    predictions_df.to_csv(file_path, index=False)


# Функция для загрузки данных из CSV файла и создания графиков
def load_predictions_from_csv(file_path):
    # Чтение данных из CSV файла
    predictions_df = pd.read_csv(file_path)

    # Создание списка для хранения фигур Plotly
    figures = []

    # Получение уникальных комбинаций самолета и позиции
    keys = predictions_df['acnum_pos'].unique()

    for key in keys:
        # Отбор данных для данной комбинации
        data = predictions_df[predictions_df['acnum_pos'] == key]
        actual_values = data['actual'].values
        predicted_values = data['predicted'].values

        # Определение индекса для отображения последних 25% реальных значений
        val_start_idx = int(0.75 * len(actual_values))

        # Создание фигуры Plotly для текущей комбинации
        fig = go.Figure()

        # Добавление реальных значений на график
        fig.add_trace(go.Scatter(x=list(range(len(actual_values))), y=actual_values,
                                 mode='lines', name=f'{key} Actual'))

        # Добавление предсказанных значений на график (только последние 25%)
        fig.add_trace(go.Scatter(x=list(range(val_start_idx, len(predicted_values))),
                                 y=predicted_values[val_start_idx:], mode='lines', name=f'{key} Predicted'))

        # Настройка заголовка графика
        fig.update_layout(title=f'{key}',
                          xaxis_title='Index', yaxis_title='EGTM Value')

        # Добавление фигуры в список
        figures.append(fig)

    return figures


# Функция для загрузки данных из файла или обучения моделей и создания графиков
def create_bayesian_ridge_plots():
    if os.path.exists(predicted_data_path):
        # Если файл существует, загружаем данные и создаем графики
        figures = load_predictions_from_csv(predicted_data_path)
    else:
        # Чтение данных из CSV файлов
        X_train = pd.read_csv(X_train_path, parse_dates=['reportts'])
        y_train = pd.read_csv(y_train_path, parse_dates=['reportts'])
        X_test = pd.read_csv(X_test_path, parse_dates=['reportts'])

        # Объединение обучающего набора признаков и целевых значений по общим колонкам и удаление строк с пропусками в 'egtm'
        dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

        # Обучение моделей для каждого из четырех графиков (по комбинациям самолета и позиции)
        fleet = ['VQ-BGU', 'VQ-BDU']
        positions = [1, 2]
        results = []

        # Создание списка для хранения фигур Plotly
        figures = []

        for acnum in fleet:
            for pos in positions:
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

                # Добавление результатов в список для сохранения в CSV
                for i in range(len(predictions)):
                    results.append([key, actual_values[i], predictions[i]])

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
                figures.append(fig)

        # Сохранение предсказанных данных в CSV файл
        save_predictions_to_csv(results, predicted_data_path)

    # Распределение графиков по сетке 2x2
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['VQ-BGU_pos_1', 'VQ-BGU_pos_2', 'VQ-BDU_pos_1', 'VQ-BDU_pos_2'])

    # Задание расположения графиков
    fig.add_trace(figures[0].data[0], row=1, col=1)
    fig.add_trace(figures[0].data[1], row=1, col=1)
    fig.add_trace(figures[1].data[0], row=2, col=1)
    fig.add_trace(figures[1].data[1], row=2, col=1)
    fig.add_trace(figures[2].data[0], row=1, col=2)
    fig.add_trace(figures[2].data[1], row=1, col=2)
    fig.add_trace(figures[3].data[0], row=2, col=2)
    fig.add_trace(figures[3].data[1], row=2, col=2)

    # Обновление общего заголовка фигуры
    fig.update_layout(title_text='EGTM Predictions')

    # Отображение фигуры
    return fig