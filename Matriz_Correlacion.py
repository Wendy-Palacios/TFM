import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from seaborn import heatmap
from matplotlib.colors import LinearSegmentedColormap


""" ---------------------------------------------------------------------------------------------------- NORMALIZACIÓN ---------------------------------------------------------------------------------------------------- """
SOC = pd.read_csv("dataset.csv", sep=',')

"""" Se separa las entradas de las salidas """
X = SOC.iloc[:,0:4]
t = SOC.iloc[:,-1].values
ndatos, nvar=X.shape
print('Número de variables: ', nvar)
print('Número de datos: ', ndatos)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

""" ---------------------------------------------------------------------------------------------------- MATRIZ DE CORRELACIÓN ---------------------------------------------------------------------------------------------------- """

def mapaCalor(matriz, etiquetas):
    rosa_gris = LinearSegmentedColormap.from_list("RosaGris", ["#E61595", "grey"])
    """ Crear el mapa de calor para la matriz de correlación """
    plt.figure(figsize=(15, 15))
    heatmap(matriz, vmin=-1, vmax=1, annot=True, cmap=rosa_gris, linewidths=1, xticklabels=etiquetas, yticklabels=etiquetas, annot_kws={"size": 14} )
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=90) 
    plt.show()


matriz = np.corrcoef(np.c_[X_scaled, t].T)
etiquetas = X.columns.tolist()
etiquetas.append("SOC")
mapaCalor(matriz, etiquetas)

