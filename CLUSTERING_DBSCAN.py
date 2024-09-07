import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from openpyxl import Workbook

# Cargar los datos
SOC = pd.read_csv("dataset.csv", sep=',')
n_features_list = [2, 3, 4]  # Diferentes números de características a probar

# Configurar la búsqueda de hiperparámetros
param_grid = {
    'eps': [0.5, 1.0, 1.1, 1.2],
    'min_samples': [3, 6, 9],  
    'n_jobs': [-1]
}

for n_features in n_features_list:
    X = SOC.iloc[:, 0:n_features]
    t = SOC.iloc[:, -1].values

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Número de entradas y variables del modelo
    ndatos, nvar = X.shape
    print(f'Número de variables: {nvar}')
    print(f'Número de datos: {ndatos}')

    best_silhouette_avg = -1
    best_params = {} 
    best_model = None
    silhouette_avgs = []

    for eps in param_grid['eps']:
        for min_samples in param_grid['min_samples']:
            try:
                print(f"Probando: eps={eps}, min_samples={min_samples}")
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)
                # Evaluar solo si hay más de un cluster
                if len(set(labels)) > 1:
                    silhouette_avg = silhouette_score(X_scaled, labels)
                    print(f"Para eps: {eps}, min_samples: {min_samples}, el coeficiente de Silhouette es: {silhouette_avg}")

                    if silhouette_avg > best_silhouette_avg:
                        best_silhouette_avg = silhouette_avg
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        best_model = dbscan
                        best_n_clusters = len(np.unique(labels[labels > -1]))
            except Exception as e:
                print(f"La combinación de parámetros no es posible: {e}")

    if best_model is not None:
        labels_optimos = best_model.labels_

        # Contar el número de elementos en cada cluster
        unique, counts = np.unique(labels_optimos, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        cluster_counts_str = ', '.join(f'Cluster {i}: {count}' for i, count in cluster_counts.items())

print("Prueba acabada")
