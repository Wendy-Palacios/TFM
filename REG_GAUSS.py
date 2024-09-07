
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import mixture

# Dataset
SOC = pd.read_csv("dataset.csv", sep=',')
SOC.iloc[:, -1] = SOC.iloc[:, -1].apply(lambda x: max(x, 0))

X = SOC.iloc[:, 0:2]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configurar y aplicar Agglomerative
gauss = mixture.GaussianMixture(
    n_components=3,
    covariance_type='tied',
    init_params='kmeans',
    random_state=50,
    tol=0.01,
    max_iter=100
)
gauss.fit(X_scaled)
labels = gauss.predict(X_scaled)

# Añadir las etiquetas de cluster al conjunto de datos
SOC['cluster'] = labels

# Dividir los diferentes DataFrames de cada cluster
SOC_cluster_1 = SOC[SOC['cluster'] == 0]
SOC_cluster_2 = SOC[SOC['cluster'] == 1]
SOC_cluster_3 = SOC[SOC['cluster'] == 2]

# Preparar entradas y salidas de cada subgrupo
X1 = SOC_cluster_1.iloc[:, 0:2]
y1 = SOC_cluster_1.iloc[:, -2].values

X2 = SOC_cluster_2.iloc[:, 0:2]
y2 = SOC_cluster_2.iloc[:, -2].values

X3 = SOC_cluster_3.iloc[:, 0:2]
y3 = SOC_cluster_3.iloc[:, -2].values

# Se utilizan diferentes modelos
name_model = ['MLP', 'KNN', 'RANDOM FOREST']

# Crear subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
posicion_x=0
posicion_y=0

# Función para entrenar y evaluar el MLP
metrics = []
def train_and_evaluate(X_train, y_train, X_test, y_test, name_model, ax, i):

    if name_model == 'MLP':
        model = MLPRegressor(max_iter=10000) 
        param_grid = {
            'hidden_layer_sizes': [(5,), (10, ), (15, ), (20, ), (25, )],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'nesterovs_momentum': [True],
            #'momentum': [0.9],
            'random_state': [42],
            'early_stopping':[True]
        }
    elif name_model == 'KNN':
        model = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': [3,10,20,80],  
            'weights': ['uniform', 'distance']
        }
    else:
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators':[100, 300, 500], 
            'max_depth': [5, 10, 15]
        }

    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=10, verbose=0, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train) 

    # Evaluar el modelo con los mejores hiperparámetros en los datos de prueba
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Graficar la salida real y la salida predicha
    ax.plot(np.arange(len(y_test)), y_test, color='blue', linewidth=2)
    ax.plot(np.arange(len(y_test)), y_pred, color='#e61595', linestyle='--', linewidth=2)

    ax.set_xlim([0, min(150, len(y_test))])

    ax.set_title(f"{name_model} Cluster {str(i)}")
    ax.set_xlabel('Muestra')
    ax.set_ylabel('SOC')

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mase = np.mean((np.abs(y_test - y_pred))/(np.mean(np.abs(np.diff(y_train)))))
    smape = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test))) #* 100
    lmls = np.mean(np.log(((((y_pred - y_test) ** 2)) + 1) / 2))

    return mse, mae, mase, smape, lmls, best_model

results_df = pd.DataFrame(columns=['Modelo', 'Nº Cluster', 'Best Params', 'MSE', 'MAE', 'MASE', 'SMAPE', 'LMLS', 'Cluster Size'])

# Entrenar y evaluar el modelo para cada clúster
for modelo in name_model:
    metrics = []
    best_params_list = []
    modelos_list = []
    n_cluster = []
    for i, (X, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)], start=1):

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        mse, mae, mase, smape, lmls, best_model = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, modelo,axs[posicion_x,posicion_y], i)
        
        results_df = results_df._append({
            'Modelo': modelo,
            'Nº Cluster': i,
            'Best Params': best_model,
            'MSE': mse,
            'MAE': mae,
            'MASE': mase,
            'SMAPE' : smape,
            'LMLS' : lmls,
            'Cluster Size': len(y),
            'Promedio': 'N/A',
            'Promedio Ponderado':  'N/A'
            }, ignore_index=True)

        metrics.append((mse, mae, mase, smape, lmls, len(y)))

        posicion_y = posicion_y + 1

    # Calcular promedios y promedios ponderados
    if metrics:
        mses, maes, mases, smapes, lmlss, sizes = zip(*metrics)
        #PROMEDIOS
        avg_mse = np.mean(mses)
        avg_mae = np.mean(maes)
        avg_mase = np.mean(mases)
        avg_smape = np.mean(smapes)
        avg_lmls = np.mean(lmlss)

        #PROMEDIOS PONDERADOS
        weighted_avg_mse = np.average(mses, weights=sizes)
        weighted_avg_mae = np.average(maes, weights=sizes)
        weighted_avg_mase = np.average(mases, weights=sizes)
        weighted_avg_smape = np.average(smapes, weights=sizes)
        weighted_avg_lmls = np.average(lmlss, weights=sizes)

        # Añadir promedios al DataFrame
        results_df = results_df._append({
            'Modelo': modelo,
            'Nº Cluster': 'PROMEDIO',
            'Best Params': 'N/A',
            'MSE': avg_mse,
            'MAE': avg_mae,
            'MASE': avg_mase,
            'SMAPE': avg_smape,
            'LMLS': avg_lmls,
            'Cluster Size': 'N/A'
        }, ignore_index=True)

        results_df = results_df._append({
            'Modelo': modelo,
            'Nº Cluster': 'PROMEDIO PONDERADO',
            'Best Params': 'N/A',
            'MSE': weighted_avg_mse,
            'MAE': weighted_avg_mae,
            'MASE': weighted_avg_mase,
            'SMAPE': weighted_avg_smape,
            'LMLS': weighted_avg_lmls,
            'Cluster Size': 'N/A'
        }, ignore_index=True)

    else:
        print("No se pudo evaluar ningún clúster.")

    posicion_y=0
    posicion_x=posicion_x+1

# Graficar
plt.tight_layout()
plt.show()