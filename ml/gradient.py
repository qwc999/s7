import itertools
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

def train_model_cv(X, y):
    assert len(X) == len(y)
    X = X.fillna(0).drop(columns=['reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'])
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    predicted = cross_val_predict(model, X, y, cv=kf)
    n_last = int(0.25 * len(y))
    y_last = y[-n_last:]
    predicted_last = predicted[-n_last:]
    rmse = np.sqrt(mean_squared_error(y_last, predicted_last))
    mae = mean_absolute_error(y_last, predicted_last)
    return rmse, mae, predicted

def save_predictions_to_csv(dataset, fleet, positions):
    predicted_data = []
    for acnum, pos in itertools.product(fleet, positions):
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
        rmse, mae, predictions = train_model_cv(X, y)
        # print(f'{acnum} pos {pos} - RMSE: {rmse}, MAE: {mae}')
        data_group = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]
        val_start_idx = int(0.75 * len(data_group))

        predicted_data.append(pd.DataFrame({
            'reportts': data_group['reportts'].iloc[val_start_idx:].reset_index(drop=True),
            'acnum': acnum,
            'pos': pos,
            'egtm': predictions[val_start_idx:]
        }))

    os.makedirs('predicted_data', exist_ok=True)
    pd.concat(predicted_data).to_csv('predicted_data/gradient.csv', index=False, columns=['reportts', 'acnum', 'pos', 'egtm'])

def create_gradient_plots():
    csv_file_path = 'predicted_data/gradient.csv'
    fleet = ['VQ-BGU', 'VQ-BDU']
    positions = [1, 2]
    if not os.path.exists(csv_file_path):
        X_train_path = './data/X_train.csv'
        y_train_path = './data/y_train.csv'
        X_train = pd.read_csv(X_train_path, parse_dates=['reportts'])
        y_train = pd.read_csv(y_train_path, parse_dates=['reportts'])

        dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])
        save_predictions_to_csv(dataset, fleet, positions)

    df = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
    predicted_df = pd.read_csv(csv_file_path, parse_dates=['reportts'])
    figures = []

    for acnum, pos in itertools.product(fleet, positions):
        key = f'{acnum}_pos_{pos}'
        data_group = df[(df['acnum'] == acnum) & (df['pos'] == pos)]
        actual_values = data_group['egtm'].values
        predicted_data = predicted_df[(predicted_df['acnum'] == acnum) & (predicted_df['pos'] == pos)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data_group['reportts'],
                y=actual_values,
                mode='lines',
                name=f'{key} Actual'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predicted_data['reportts'],
                y=predicted_data['egtm'],
                mode='lines',
                name=f'{key} Predicted'
            )
        )
        fig.update_layout(
            title=f'{key}',
            xaxis_title='Date',
            yaxis_title='EGTM Value'
        )
        figures.append(fig)

    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(figures[0].data[0], row=1, col=1)
    fig.add_trace(figures[0].data[1], row=1, col=1)
    fig.add_trace(figures[1].data[0], row=2, col=1)
    fig.add_trace(figures[1].data[1], row=2, col=1)
    fig.add_trace(figures[2].data[0], row=1, col=2)
    fig.add_trace(figures[2].data[1], row=1, col=2)
    fig.add_trace(figures[3].data[0], row=2, col=2)
    fig.add_trace(figures[3].data[1], row=2, col=2)

    fig.update_layout(title_text='EGTM Predictions')

    return fig