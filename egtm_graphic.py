"""
код показывает график egtm
для обучения не подходит т к нет разбиания на pos1 pos2
"""
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/y_train.csv')
egtm_column_name = df.columns[3]
acnum_column_name = 'acnum'

# индекс первого появления VQ-BDU
acnum_change_index = df[df[acnum_column_name] == 'VQ-BDU'].index[0]

plt.figure(figsize=(10, 6))
plt.plot(df[egtm_column_name][:acnum_change_index], label='VQ-BGU', color='blue')
plt.plot(df[egtm_column_name][acnum_change_index:], label='VQ-BDU', color='orange')
plt.xlabel('Индекс')
plt.ylabel(egtm_column_name)
plt.legend()
plt.grid(True)
plt.show()
