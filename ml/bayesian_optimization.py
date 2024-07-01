import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real
import os


def load_data(X_train_path='./data/X_train.csv', y_train_path='./data/y_train.csv', X_test_path='./data/X_test.csv'):
    """Загрузка данных из CSV файлов."""
    X_train = pd.read_csv(X_train_path, parse_dates=['reportts'])
    y_train = pd.read_csv(y_train_path, parse_dates=['reportts'])
    X_test = pd.read_csv(X_test_path, parse_dates=['reportts'])
    return X_train, y_train, X_test


def train_model_cv(X, y):
    """Обучение модели с кросс-валидацией и оптимизацией гиперпараметров."""
    assert len(X) == len(y)
    X = X.fillna(0).drop(columns=['reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'])

    search_spaces = {
        'alpha_1': Real(1e-6, 1e3, prior='log-uniform'),
        'alpha_2': Real(1e-6, 1e3, prior='log-uniform'),
        'lambda_1': Real(1e-6, 1e3, prior='log-uniform'),
        'lambda_2': Real(1e-6, 1e3, prior='log-uniform'),
        'tol': Real(1e-6, 1e-1, prior='log-uniform')
    }

    opt_model = BayesSearchCV(
        estimator=BayesianRidge(),
        search_spaces=search_spaces,
        n_iter=100,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        n_points=5
    )

    opt_model.fit(X, y)

    best_params = opt_model.best_params_
    model = opt_model.best_estimator_

    predicted = opt_model.predict(X)

    rmse = mean_squared_error(y, predicted, squared=False)
    mae = mean_absolute_error(y, predicted)

    return rmse, mae, predicted, best_params


def load_predictions_from_csv(file_path):
    """Загрузка предсказанных данных из CSV файла."""
    return pd.read_csv(file_path)


def plot_predictions(predictions_df, dataset):
    """Построение графиков из предсказанных данных."""
    figures = []

    for index, row in predictions_df.iterrows():
        key = f"{row['acnum']}_pos_{row['pos']}"
        actual_values = dataset[(dataset['acnum'] == row['acnum']) & (dataset['pos'] == row['pos'])]['egtm'].values
        predictions = eval(row['predictions'])

        val_start_idx = int(0.75 * len(actual_values))

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=list(range(len(actual_values))), y=actual_values,
                                 mode='lines', name=f'{key} Actual'))

        fig.add_trace(go.Scatter(x=list(range(val_start_idx, len(predictions))),
                                 y=predictions[val_start_idx:], mode='lines', name=f'{key} Predicted'))

        fig.update_layout(title=f'{key} RMSE={row["rmse"]:.3f} MAE={row["mae"]:.3f}',
                          xaxis_title='Index', yaxis_title='EGTM Value')

        figures.append(fig)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['VQ-BGU_pos_1', 'VQ-BGU_pos_2', 'VQ-BDU_pos_1', 'VQ-BDU_pos_2'])

    fig.add_trace(figures[0].data[0], row=1, col=1)
    fig.add_trace(figures[0].data[1], row=1, col=1)
    fig.add_trace(figures[1].data[0], row=2, col=1)
    fig.add_trace(figures[1].data[1], row=2, col=1)
    fig.add_trace(figures[2].data[0], row=1, col=2)
    fig.add_trace(figures[2].data[1], row=1, col=2)
    fig.add_trace(figures[3].data[0], row=2, col=2)
    fig.add_trace(figures[3].data[1], row=2, col=2)

    fig.update_layout(title_text='EGTM Predictions')

    fig.show()


def perform_bayesian_optimization(X_train, y_train, predictions_file_path='./predicted_data/bayesianoptimization.csv'):
    """Выполнение байесовской оптимизации с сохранением и загрузкой результатов."""
    if os.path.exists(predictions_file_path):
        predictions_df = load_predictions_from_csv(predictions_file_path)
        print("Загрузка предсказанных данных из файла.")
        dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])
    else:
        dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

        fleet = ['VQ-BGU', 'VQ-BDU']
        positions = [1, 2]
        results = []

        for acnum in fleet:
            for pos in positions:
                key = f'{acnum}_pos_{pos}'
                X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
                y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
                rmse, mae, predictions, best_params = train_model_cv(X, y)

                results.append({
                    'acnum': acnum,
                    'pos': pos,
                    'rmse': rmse,
                    'mae': mae,
                    'predictions': str(list(predictions)),
                    'best_params': best_params
                })

        predictions_df = pd.DataFrame(results)
        predictions_df.to_csv(predictions_file_path, index=False)
        print(f"Предсказанные данные сохранены в файл: {predictions_file_path}")

    plot_predictions(predictions_df, dataset)


# Пример использования функции
if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    perform_bayesian_optimization(X_train, y_train)
