import numpy as np
import pandas as pd
from scipy.spatial import distance

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

puntos = {
    "Punto A": (2, 3),
    "Punto B": (5, 4),
    "Punto C": (1, 1),
    "Punto D": (6, 7),
    "Punto E": (3, 5),
    "Punto F": (8, 2),
    "Punto G": (4, 6),
    "Punto H": (2, 1)
}

nombres = list(puntos.keys())
coords = np.array(list(puntos.values()))

df_coords = pd.DataFrame(coords, columns=["X", "Y"], index=nombres)
print("Coordenadas de los puntos: \n", df_coords)

def crear_matriz_distancia(metric):
    dist = distance.cdist(coords, coords, metric=metric)
    return pd.DataFrame(dist, index=nombres, columns=nombres)

euclidiana = crear_matriz_distancia("euclidean")
manhattan = crear_matriz_distancia("cityblock")
chebyshev = crear_matriz_distancia("chebyshev")

print("\nDistancia Euclidiana entre cada uno de los puntos:")
print(euclidiana.round(6))  

print("\nDistancia Manhatthan entre cada uno de los puntos:")
print(manhattan.round(1))  

print("\nDistancia Chebyshev entre cada uno de los puntos:")
print(chebyshev.round(1))  

def extremos(df, nombre):
    df_sin_diag = df.where(~np.eye(df.shape[0], dtype=bool))
    min_val = df_sin_diag.min().min()
    max_val = df_sin_diag.max().max()
    min_pos = df_sin_diag.stack().idxmin()
    max_pos = df_sin_diag.stack().idxmax()
    print(f"\nPuntos más cercanos según {nombre}: {min_pos} con distancia {min_val:.6f}")
    print(f"Puntos más lejanos según {nombre}: {max_pos} con distancia {max_val:.6f}")

extremos(euclidiana, "Euclidiana")
extremos(manhattan, "Manhattan")
extremos(chebyshev, "Chebyshev")
