import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Загрузка данных
X_train = pd.read_csv('./data/X_train.csv', parse_dates=['reportts'])
y_train = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
X_test = pd.read_csv('./data/X_test.csv', parse_dates=['reportts'])
dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

# Определение модели нейросети
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model_pytorch(X, y):
    assert len(X) == len(y)

    # Предобработка данных
    X = X.fillna(0).drop(columns=['reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'])
    y = y.fillna(0)

    # Нормализация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    # Преобразование данных в тензоры
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Создание модели
    model = SimpleNN(input_dim=X_train.shape[1])

    # Определение функции потерь и оптимизатора
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Оценка модели
    model.eval()
    with torch.no_grad():
        predicted = model(X_val_tensor).numpy()
        rmse = mean_squared_error(y_val, predicted, squared=False)
        mae = mean_absolute_error(y_val, predicted)

    return rmse, mae, model, X_val_tensor, predicted

# Обучение моделей для каждого из четырех графиков
fleet = ['VQ-BGU', 'VQ-BDU']
positions = [1, 2]
results = {}

for acnum in fleet:
    for pos in positions:
        key = f'{acnum}_pos_{pos}'
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])
        y = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)]['egtm']
        rmse, mae, model, indices, predictions = train_model_pytorch(X, y)
        results[key] = (rmse, mae, indices, predictions)
        print(f'{key} RMSE={rmse:.3f} MAE={mae:.3f}')

# Чтение данных для графиков
df = pd.read_csv('data/y_train.csv')
egtm_column_name = df.columns[3]
acnum_column_name = 'acnum'
pos_column_name = df.columns[2]

# Построение графиков
plt.figure(figsize=(14, 10))

for i, (label, group) in enumerate(results.items(), 1):
    rmse, mae, indices, predictions = group
    data_group = df[(df[acnum_column_name] == label.split('_')[0]) & (df[pos_column_name] == int(label.split('_')[2]))]

    # Отображение всех реальных значений
    plt.subplot(2, 2, i)
    plt.plot(data_group[egtm_column_name].values, label=f'{label} Actual')

    # Выделение последних 25% реальных значений для совпадения с предсказанными
    val_start_idx = len(data_group) - len(predictions)
    plt.plot(range(val_start_idx, len(data_group)), predictions, label=f'{label} Predicted', color='orange')

    plt.title(f'{label} RMSE={rmse:.3f} MAE={mae:.3f}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
