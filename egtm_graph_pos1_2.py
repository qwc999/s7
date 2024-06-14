import pandas as pd
import matplotlib.pyplot as plt

# Чтение данных
df = pd.read_csv('data/y_train.csv')
egtm_column_name = df.columns[3]
acnum_column_name = 'acnum'
pos_column_name = df.columns[2]

# Сглаживание
# window_size = 10  # Размер окна для скользящего среднего
# df['egtm_smoothed'] = df[egtm_column_name].rolling(window=window_size, center=True).mean()

# Разделение данных на группы по acnum и pos
groups = {
    'VQ-BGU_pos_1': df[(df[acnum_column_name] == 'VQ-BGU') & (df[pos_column_name] == 1)],
    'VQ-BGU_pos_2': df[(df[acnum_column_name] == 'VQ-BGU') & (df[pos_column_name] == 2)],
    'VQ-BDU_pos_1': df[(df[acnum_column_name] == 'VQ-BDU') & (df[pos_column_name] == 1)],
    'VQ-BDU_pos_2': df[(df[acnum_column_name] == 'VQ-BDU') & (df[pos_column_name] == 2)]
}

# Построение графиков
plt.figure(figsize=(14, 10))

for i, (label, group) in enumerate(groups.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(group[egtm_column_name], label=label)
    plt.title(f'{label}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
