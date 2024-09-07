import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from itertools import product

# Cargar los datos
SOC = pd.read_csv("dataset.csv", sep=',')
n_features_list = [2, 3, 4]  # Diferentes números de características a probar

# Definir los posibles valores para cada hiperparámetro
n_clusters_list = range(2, 10)
eigen_solvers = ['arpack', 'lobpcg', 'amg']
gamma_list = [0.1, 1, 10]
assign_labels_list = ['kmeans', 'discretize','cluster_qr']
affinities = ['nearest_neighbors', 'rbf']
random_state = 42
n_neighborss = [850, 1000, 1200]  


for n_features in n_features_list:
    X = SOC.iloc[:, 0:n_features]
    
    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Número de entradas y variables del modelo
    ndatos, nvar = X.shape
    print(f'Número de variables: {nvar}')
    print(f'Número de datos: {ndatos}')

    best_params = None
    best_silhouette_avg = -1
    best_n_clusters = None

    # Búsqueda manual con cross-validation
    for n_clusters, eigen_solver, assign_labels, affinity in product(n_clusters_list, eigen_solvers, assign_labels_list, affinities):
        if affinity == 'rbf':
            param_grid = [{'gamma': gamma} for gamma in gamma_list]
        else:  # affinity == 'nearest_neighbors'
            param_grid = [{'n_neighbors': n_neighbors} for n_neighbors in n_neighborss]

        for params in param_grid:

            # Crear el modelo con los hiperparámetros actuales
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                eigen_solver=eigen_solver,
                affinity=affinity,
                assign_labels=assign_labels,
                random_state=random_state,
                n_jobs=-1,
                **params  # Añadir parámetros específicos de 'gamma' o 'n_neighbors'
            )

            # Ajustar el modelo y predecir en el conjunto de test
            labels = spectral.fit_predict(X_scaled)

            # Calcular la métrica de evaluación
            silhouette_avg = silhouette_score(X_scaled, labels)

            # Actualizar los mejores parámetros
            if silhouette_avg > best_silhouette_avg:
                best_silhouette_avg = silhouette_avg
                best_params = {
                    'n_clusters': n_clusters,
                    'eigen_solver': eigen_solver,
                    'affinity': affinity,
                    'assign_labels': assign_labels,
                    'random_state': random_state,
                    **params
                }
                best_n_clusters = n_clusters


    # Evaluar el coeficiente de Silhouette para diferentes valores de k
    k_values = range(2, 10)
    silhouette_avgs = []
    for k in k_values:
        modelo = SpectralClustering(n_clusters=k, **{key: best_params[key] for key in best_params if key != 'n_clusters'}, random_state=42)
        labels = modelo.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, labels)
        silhouette_avgs.append(silhouette_avg)

    # Encontrar el número de clusters con el coeficiente de Silhouette más alto
    optimal_k = k_values[np.argmax(silhouette_avgs)]

    # Ajustar el modelo Spectral con el número óptimo de clusters
    modelo_optimo = SpectralClustering(n_clusters=optimal_k, **{key: best_params[key] for key in best_params if key != 'n_clusters'}, random_state=random_state)
    labels_optimos = modelo_optimo.fit_predict(X_scaled)

    # Contar el número de elementos en cada cluster
    unique, counts = np.unique(labels_optimos, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    cluster_counts_str = ', '.join(f'Cluster {i}: {count}' for i, count in cluster_counts.items())

print("Prueba acabada")
