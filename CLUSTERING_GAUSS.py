
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV

# Cargar los datos
SOC = pd.read_csv("dataset.csv", sep=',')
n_features_list = [2,3,4]  # Diferentes números de características a probar

# Configurar la búsqueda en cuadrícula
param_grid = {
    'n_components': range(2, 10), 
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  
    'init_params': ['kmeans','k-means++','random','random_from_data'],
    'max_iter': [100],
    'tol': [1e-2],
    'random_state': [42]
}

# Definir función de scoring personalizada
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)  
    score = silhouette_score(X, labels) 
    return score

# Crear el objeto de scorer para GridSearchCV
scorer = make_scorer(silhouette_scorer, greater_is_better=True)

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

    # Grid Search
    gmm = GaussianMixture(random_state=10)
    grid_search = GridSearchCV(gmm, param_grid=param_grid, scoring=silhouette_scorer, cv=10, n_jobs=-1)
    grid_search.fit(X_scaled)
    best_params = grid_search.best_params_
    best_n_components = best_params['n_components']
    print(f"Mejores hiperparámetros para {n_features} características: {best_params}")
    print(f'Nº de componentes óptimo usando grid search: {best_n_components}')

    # Evaluar el coeficiente de Silhouette para diferentes valores de n_components
    n_components_values = range(2, 10)
    silhouette_avgs = []
    for n_components in n_components_values:
        modelo = GaussianMixture(n_components=n_components, **{key: best_params[key] for key in best_params if key != 'n_components'})
        labels = modelo.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, labels)
        silhouette_avgs.append(silhouette_avg)
        print(f'Para n_components = {n_components}, el coeficiente de Silhouette es {silhouette_avg}')

    # Encontrar el número de componentes con el coeficiente de Silhouette más alto
    optimal_n_components = n_components_values[np.argmax(silhouette_avgs)]
    print(f'El número óptimo de componentes según el coeficiente de Silhouette es: {optimal_n_components}')

    # Ajustar el modelo GaussianMixture con el número óptimo de componentes
    modelo_optimo = GaussianMixture(n_components=optimal_n_components, **{key: best_params[key] for key in best_params if key != 'n_components'})
    labels_optimos = modelo_optimo.fit_predict(X_scaled)

    # Contar el número de elementos en cada cluster
    unique, counts = np.unique(labels_optimos, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    cluster_counts_str = ', '.join(f'Cluster {i}: {count}' for i, count in cluster_counts.items())

print("Prueba acabada'")
