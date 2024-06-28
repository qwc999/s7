import itertools
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def train_model_cv(X, y):
    assert len(X) == len(y)
    X = X.fillna(0).drop(columns=['reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'])
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predicted = cross_val_predict(model, X, y, cv=kf)
    rmse = np.sqrt(mean_squared_error(y, predicted))
    mae = mean_absolute_error(y, predicted)
    return rmse, mae, predicted

def create_gradient_plots():
    X_train_path = './data/X_train.csv'
    y_train_path = './data/y_train.csv'
    X_test_path = './data/X_test.csv'
    X_train = pd.read_csv(X_train_path, parse_dates=['reportts'])
    y_train = pd.read_csv(y_train_path, parse_dates=['reportts'])
    X_test = pd.read_csv(X_test_path, parse_dates=['reportts'])

    dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

    fleet = ['VQ-BGU', 'VQ-BDU']
    positions = [1, 2]

    figures = []

    for acnum, pos in itertools.product(fleet, positions):
        key = f'{acnum}_pos_{pos}'
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
        rmse, mae, predictions = train_model_cv(X, y)

        data_group = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]
        actual_values = data_group['egtm'].values
        val_start_idx = int(0.75 * len(actual_values))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(actual_values))), 
                y=actual_values,
                mode='lines', 
                name=f'{key} Actual'
                )
            )
        fig.add_trace(
            go.Scatter(
                x=list(range(val_start_idx, len(predictions))),
                y=predictions[val_start_idx:], 
                mode='lines', 
                name=f'{key} Predicted')
            )
        fig.update_layout(
            title=f'{key} RMSE={rmse:.3f} MAE={mae:.3f}',
            xaxis_title='Index', 
            yaxis_title='EGTM Value')
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

    return fig
