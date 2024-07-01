"""
Кластеризация с использованием KMeans
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Function to perform clustering and assign cluster labels
def perform_clustering(X):
    X = X.fillna(0).drop(columns=[
        'reportts', 'acnum', 'pos', 'fltdes', 'dep', 'arr'
    ])
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)

    return kmeans, cluster_labels, silhouette_avg


# Reading data
X_train = pd.read_csv('./data/X_train.csv', parse_dates=['reportts'])
y_train = pd.read_csv('./data/y_train.csv', parse_dates=['reportts'])
X_test = pd.read_csv('./data/X_test.csv', parse_dates=['reportts'])

dataset = X_train.merge(y_train, on=['acnum', 'pos', 'reportts']).dropna(subset=['egtm'])

fleet = ['VQ-BGU', 'VQ-BDU']
positions = [1, 2]
results = {}

# DataFrame to store results
predictions_df = pd.DataFrame(columns=['reportts', 'acnum', 'pos', 'cluster_label'])

for acnum in fleet:
    for pos in positions:
        key = f'{acnum}_pos_{pos}'
        X = dataset[(dataset['acnum'] == acnum) & (dataset['pos'] == pos)].drop(columns=['egtm'])

        kmeans, cluster_labels, silhouette_avg = perform_clustering(X)

        # Append cluster labels to the DataFrame
        temp_df = pd.DataFrame({
            'reportts': X['reportts'],
            'acnum': acnum,
            'pos': pos,
            'cluster_label': cluster_labels
        })
        predictions_df = pd.concat([predictions_df, temp_df], ignore_index=True)

        results[key] = (silhouette_avg, cluster_labels)
        print(f'{key} Silhouette Score={silhouette_avg:.3f}')

# Save cluster labels to a CSV file
predictions_df.to_csv('predicted_data/kmeans_clusters.csv', index=False)

# Reading data for plots
df = pd.read_csv('data/y_train.csv')
egtm_column_name = df.columns[3]
acnum_column_name = 'acnum'
pos_column_name = df.columns[2]

# Plotting graphs
plt.figure(figsize=(14, 10))

for i, (label, group) in enumerate(results.items(), 1):
    silhouette_avg, cluster_labels = group
    data_group = df[(df[acnum_column_name] == label.split('_')[0]) & (df[pos_column_name] == int(label.split('_')[2]))]

    plt.subplot(2, 2, i)
    plt.scatter(data_group.index, data_group[egtm_column_name], c=cluster_labels, cmap='viridis',
                label=f'{label} Clusters')

    plt.title(f'{label} Silhouette Score={silhouette_avg:.3f}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
