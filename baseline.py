import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error


X_train = pd.read_csv('./data/X_train.csv', parse_dates=['reportts'])
y_train = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
X_test = pd.read_csv('./data/X_test.csv', parse_dates=['reportts'])
dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])


def train_model(X, y):
    assert len(X) == len(y)
    X = X.fillna(0).drop(columns=[
      'reportts', 'acnum', 'pos', 'fltdes',	'dep', 'arr'
    ])
    # обучаем на более ранних данных, тестируем на поздних
    train_i = int(len(X) * 75 / 100)
    X_train, y_train = X[0:train_i], y[0:train_i]
    X_val, y_test = X[train_i:], y[train_i:]

    model = Ridge(alpha=5)
    model.fit(X_train, y_train)

    predicted = model.predict(X_val)
    rmse = mean_squared_error(y_test, predicted, squared=False)
    mae = mean_absolute_error(y_test, predicted)

    for pred, true in zip(predicted, y_test):
        print(f'Predicted: {pred:.3f}, True: {true:.3f}')

    return rmse, mae, model


fleet = ['VQ-BGU', 'VQ-BDU']
for acnum in fleet:
    X = dataset[dataset['acnum'] == acnum].drop(columns=['egtm'])
    y = dataset[dataset['acnum'] == acnum]['egtm']
    print(y)
    rmse, mae, model = train_model(X, y)
    print(f'acnum={acnum} RMSE={rmse:.3f} MAE={mae:.3f}')