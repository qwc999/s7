"""
Регуляризованная линейная модель (Ridge)
"""
import itertools
import pandas as pd
import plotly.graph_objects as go  
from plotly.subplots import make_subplots 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Function to train the model and predict values
def train_model(X, y):
    assert len(X) == len(y)
    X = X.fillna(0).drop(columns=[
        'reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'
    ])
    train_i = int(len(X) * 75 / 100)
    X_train, y_train = X[:train_i], y[:train_i]
    X_val, y_test = X[train_i:], y[train_i:]

    model = Ridge(alpha=5)
    model.fit(X_train, y_train)

    predicted = model.predict(X_val)
    rmse = mean_squared_error(y_test, predicted, squared=False)
    mae = mean_absolute_error(y_test, predicted)

    return rmse, mae, model, X_val.index, predicted


def create_ridge_plots():
    # Reading data

    X_train = pd.read_csv('./data/X_train.csv', parse_dates=['reportts'])
    y_train = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
    X_test = pd.read_csv('./data/X_test.csv', parse_dates=['reportts'])

    dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])
    # Training models for each of the four graphs
    fleet = ['VQ-BGU', 'VQ-BDU']
    positions = [1, 2]
    results = {}

    # DataFrame to store results
    predictions_df = pd.DataFrame(columns=['reportts', 'acnum', 'pos', 'egtm'])

    for acnum, pos in itertools.product(fleet, positions):
        key = f'{acnum}_pos_{pos}'
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
        rmse, mae, model, indices, predictions = train_model(X, y)
        # Append predictions to the DataFrame
        temp_df = pd.DataFrame({
            'reportts': X.loc[indices, 'reportts'],
            'acnum': acnum,
            'pos': pos,
            'egtm': predictions
        })
        predictions_df = pd.concat([predictions_df, temp_df], ignore_index=True)

        # print(predictions)
        results[key] = (rmse, mae, indices, predictions)
        print(f'{key} RMSE={rmse:.3f} MAE={mae:.3f}')

    # Save predictions to a CSV file
    predictions_df.to_csv('predicted_data/ridge.csv', index=False)

    # Reading data for plots
    df = pd.read_csv('data/y_train.csv')
    egtm_column_name = df.columns[3]
    acnum_column_name = 'acnum'
    pos_column_name = df.columns[2]

    # Plotting graphs
    graphs = []

    for i, (label, group) in enumerate(results.items(), 1):
        rmse, mae, indices, predictions = group
        data_group = df[(df[acnum_column_name] == label.split('_')[0]) & (df[pos_column_name] == int(label.split('_')[2]))]
        val_start_idx = len(data_group) - len(predictions)
        # Plot all actual values
        
        graphs.append(
            [
                f'{label} RMSE={rmse:.3f} MAE={mae:.3f}',
                go.Scatter(
                    y=data_group[egtm_column_name].values, 
                    name=f'{label} Actual'
                ),
                go.Scatter(
                    x=list(range(val_start_idx, len(data_group))),
                    y=predictions,
                    name=f'{label} Predicted',
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