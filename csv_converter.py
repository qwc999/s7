"""
для конвертации csv файлов
функция принимает список столбцов для будущего файла и новое имя файла
возвращает
"""
import pandas as pd


def csv_convert(columns, output_file="data/newY_train.csv"):
    X_train = pd.read_csv('./data/X_train.csv', parse_dates=['reportts'])
    y_train = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
    dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])
    dataset = dataset[columns]
    dataset.to_csv(output_file, index=False)
    return dataset
