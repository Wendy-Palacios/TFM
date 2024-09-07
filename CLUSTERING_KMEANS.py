import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV

# Cargar los datos
SOC = pd.read_csv("dataset.csv", sep=',')
n_features_list = [2,3,4]  # Diferentes números de características a probar

# Configurar grid search params
param_grid = {
    'n_clusters': range(2, 10), 
    'init': ['random', 'k-means++'],
    'n_init': [50],
    'max_iter': [100, 1000, 1500],
    'tol': [1e-2],
    'random_state': [42],
    'algorithm': ['lloyd', 'elkan']
}

# Definir función de scoring
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)  
    score = silhouette_score(X, labels)  
    return score

# Crear el objeto de scorer para GridSearchCV
scorer = make_scorer(silhouette_scorer, greater_is_better=True)

for n_features in n_features_list:
    X = SOC.iloc[:, 0:n_features]
    print(X)
    t = SOC.iloc[:, -1].values

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Número de entradas y variables del modelo
    ndatos, nvar = X.shape
    print(f'Número de variables: {nvar}')
    print(f'Número de datos: {ndatos}')

    # Grid Search
    kmeans = cluster.KMeans(random_state=10)
    grid_search = GridSearchCV(kmeans, param_grid=param_grid, scoring=silhouette_scorer, cv=10, n_jobs=-1)
    grid_search.fit(X_scaled)
    best_params = grid_search.best_params_
    best_n_clusters = best_params['n_clusters']

    # Evaluar el coeficiente de Silhouette para diferentes valores de k
    k_values = range(2, 10)
    silhouette_avgs = []
    for k in k_values:
        modelo = cluster.KMeans(n_clusters=k, **{key: best_params[key] for key in best_params if key != 'n_clusters'})
        modelo.fit(X_scaled)
        labels = modelo.labels_
        silhouette_avg = silhouette_score(X_scaled, labels)
        silhouette_avgs.append(silhouette_avg)
        print(f'Para k = {k}, el coeficiente de Silhouette es {silhouette_avg}')

    # Encontrar el número de clusters con el coeficiente de Silhouette más alto
    optimal_k = k_values[np.argmax(silhouette_avgs)]
    
    # Ajustar el modelo KMeans con el número óptimo de clusters
    modelo_optimo = cluster.KMeans(n_clusters=optimal_k, **{key: best_params[key] for key in best_params if key != 'n_clusters'})
    modelo_optimo.fit(X_scaled)
    labels_optimos = modelo_optimo.labels_

    # Contar el número de elementos en cada cluster
    unique, counts = np.unique(labels_optimos, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    cluster_counts_str = ', '.join(f'Cluster {i}: {count}' for i, count in cluster_counts.items())

print("Prueba acabada'")
