from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from itertools import product

# Cargar los datos
SOC = pd.read_csv("03 ETAPA 1.1/my_data_V01.csv", sep=',')
n_features_list = [2, 3, 4]  # Diferentes números de características a probar

n_clusters_range = range(2, 10)  
metrics = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
linkages = ['ward', 'complete', 'average', 'single']
compute_full_trees = [True, False]
n_splits = 5  # Número de folds para cross-validation

for n_features in n_features_list:
    X = SOC.iloc[:, 0:n_features]

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Número de entradas y variables del modelo
    ndatos, nvar = X.shape
    print(f'Número de variables: {nvar}')
    print(f'Número de datos: {ndatos}')

    best_score = -np.inf
    best_params = None
    best_n_clusters = None

    # Búsqueda manual
    for n_clusters, metric, linkage, compute_full_tree in product(n_clusters_range, metrics, linkages, compute_full_trees):
        try:
            agglomerative = cluster.AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage, compute_full_tree=compute_full_tree)
            labels = agglomerative.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, labels)
                        
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_params = {'n_clusters': n_clusters, 'metric': metric, 'linkage': linkage, 'compute_full_tree': compute_full_tree}
                best_n_clusters = n_clusters
        except Exception as e:
            print(f'Error con n_clusters={n_clusters}, metric={metric}, linkage={linkage}, compute_full_tree={compute_full_tree}: {e}')

    # Evaluar el coeficiente de Silhouette para diferentes valores de k
    k_values = range(2, 10)
    silhouette_avgs = []
    for k in k_values:
        modelo = cluster.AgglomerativeClustering(n_clusters=k, **{key: best_params[key] for key in best_params if key != 'n_clusters'})
        modelo.fit(X_scaled)
        labels = modelo.labels_
        silhouette_avg = silhouette_score(X_scaled, labels)
        silhouette_avgs.append(silhouette_avg)

    # Encontrar el número de clusters con el coeficiente de Silhouette más alto
    optimal_k = k_values[np.argmax(silhouette_avgs)]
    
    # Ajustar el modelo Agglomerative con el número óptimo de clusters
    modelo_optimo = cluster.AgglomerativeClustering(n_clusters=optimal_k, **{key: best_params[key] for key in best_params if key != 'n_clusters'})
    modelo_optimo.fit(X_scaled)
    labels_optimos = modelo_optimo.labels_

    # Contar el número de elementos en cada cluster
    unique, counts = np.unique(labels_optimos, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    cluster_counts_str = ', '.join(f'Cluster {i}: {count}' for i, count in cluster_counts.items())

print("Prueba acabada")
