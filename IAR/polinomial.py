import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.ndimage import median_filter


# Function to train the model and predict values
def train_model(X, y, degree=2, alpha=0.1):
    assert len(X) == len(y)
    X = X.fillna(0).drop(columns=[
        'reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'
    ])

    indices = X.index
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Split data
    X_train, X_val, y_train, y_test, indices_train, indices_val = train_test_split(X_poly, y, indices, test_size=0.25,
                                                                                   random_state=42)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    predicted = model.predict(X_val)
    smoothed_predicted = median_filter(predicted, size=3)

    rmse = mean_squared_error(y_test, smoothed_predicted, squared=False)
    mae = mean_absolute_error(y_test, smoothed_predicted)

    return rmse, mae, model, indices_val, smoothed_predicted


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

for acnum in fleet:
    for pos in positions:
        key = f'{acnum}_pos_{pos}'
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
        rmse, mae, model, indices_val, predictions = train_model(X, y)
        # Append predictions to the DataFrame
        temp_df = pd.DataFrame({
            'reportts': X.loc[indices_val, 'reportts'],
            'acnum': acnum,
            'pos': pos,
            'egtm': predictions
        })
        predictions_df = pd.concat([predictions_df, temp_df], ignore_index=True)

        print(predictions)
        results[key] = (rmse, mae, indices_val, predictions)
        print(f'{key} RMSE={rmse:.3f} MAE={mae:.3f}')

# Save predictions to a CSV file
predictions_df.to_csv('predicted_data/ridge_regression.csv', index=False)

# Reading data for plots
df = pd.read_csv('data/y_train.csv')
egtm_column_name = df.columns[3]
acnum_column_name = 'acnum'
pos_column_name = df.columns[2]

# Plotting graphs
plt.figure(figsize=(14, 10))

for i, (label, group) in enumerate(results.items(), 1):
    rmse, mae, indices_val, predictions = group
    data_group = df[(df[acnum_column_name] == label.split('_')[0]) & (df[pos_column_name] == int(label.split('_')[2]))]

    # Plot all actual values
    plt.subplot(2, 2, i)
    plt.plot(data_group[egtm_column_name].values, label=f'{label} Actual')

    # Highlight the last 25% actual values to match the predicted values
    val_start_idx = len(data_group) - len(predictions)
    plt.plot(range(val_start_idx, len(data_group)), predictions, label=f'{label} Predicted', color='orange')

    plt.title(f'{label} RMSE={rmse:.3f} MAE={mae:.3f}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
