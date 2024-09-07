import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Dataset
SOC = pd.read_csv("dataset.csv", sep=',')
SOC.iloc[:, -1] = SOC.iloc[:, -1].apply(lambda x: max(x, 0))

X1 = SOC.iloc[:, 0:2]
y1 = SOC.iloc[:, -1].values

# Se utilizan diferentes modelos
name_model = ['MLP', 'KNN', 'RANDOM FOREST']
posicion=0

# Crear subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 11))
# Función para entrenar y evaluar el MLP
def train_and_evaluate(X_train, y_train, X_test, y_test, name_model):

    if name_model == 'MLP':

        model = MLPRegressor(max_iter=10000)
        param_grid = {
            'hidden_layer_sizes': [(10, 10), (15,15),(20,20), (25,25)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'nesterovs_momentum': [True],
            'momentum': [0.9],
            'random_state': [42],
            'early_stopping':[False],#[False, True]
            'warm_start':[False]# [False, True]
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

    # Imprimir los mejores hiperparámetros encontrados
    print(f"Mejores hiperparámetros para el clúster: ", grid_search.best_params_)
  
    # Cálculo de errores
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mase = np.mean((np.abs(y_test - y_pred))/(np.mean(np.abs(np.diff(y_train)))))
    smape = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test))) #* 100
    lmls = np.mean(np.log(((((y_pred - y_test) ** 2)) + 1) / 2))

    return mse, mae, mase, smape, lmls, best_model, y_pred

results_df = pd.DataFrame(columns=['Modelo', 'Nº Inputs', 'Best Params', 'MSE', 'MAE', 'MASE', 'SMAPE', 'LMLS', 'Cluster Size'])
for modelo in name_model:
    metrics = []
    best_params_list = []
    n_entradas = []
    mse_now = -1
    
    for i, (X, y) in enumerate([(X1, y1)], start=1):

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

        scaler=StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mse, mae, mase, smape, lmls, best_model, y_pred = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test,modelo)

        metrics.append((mse, mae, mase, smape, lmls, len(y)))

        #Graficar
        if mse > mse_now: 
            ax=axs[posicion]
            ax.plot(np.arange(len(y_test)), y_test, color='blue', linewidth=2)
            ax.plot(np.arange(len(y_test)), y_pred, color='#e61595', linestyle='--', linewidth=2)

            ax.set_xlim([0, min(150, len(y_test))])
            ax.set_title(f"{modelo} Nº Inputs {str(i+1)}")
            ax.set_xlabel('Muestra')
            ax.set_ylabel('SOC')
            mse_now = mse

        results_df = results_df._append({
        'Modelo': modelo,
        'Nº Inputs': i+1,
        'Best Params': best_model,
        'MSE': mse,
        'MAE': mae,
        'MASE': mase,
        'SMAPE' : smape,
        'LMLS' : lmls,
        'Cluster Size': len(y)
    }, ignore_index=True)
        
    posicion=posicion+1