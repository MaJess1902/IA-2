import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 1. Dataset ampliado con 10 clientes y 3 dimensiones: [Edad, Salario, Num_Hijos]
X_entrenamiento = np.array([
    [20, 30, 0],  # Cliente 1
    [40, 50, 2],  # Cliente 2
    [35, 45, 1],  # Cliente 3
    [18, 22, 0],  # Cliente 4
    [22, 28, 0],  # Cliente 5
    [50, 80, 3],  # Cliente 6
    [45, 75, 2],  # Cliente 7
    [28, 35, 1],  # Cliente 8
    [60, 90, 1],  # Cliente 9
    [25, 32, 0],  # Cliente 10
])

# Etiquetas: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])

# Punto nuevo a consultar: [Edad: 30, Salario: 40, Num_Hijos: 1]
nuevo_cliente = np.array([[30, 40, 1]])

# Experimentación con K = 1
modelo_k1 = KNeighborsClassifier(n_neighbors=1)
modelo_k1.fit(X_entrenamiento, Y_entrenamiento)
prediccion_k1 = modelo_k1.predict(nuevo_cliente)
print("Predicción con K=1:", "COMPRA (1)" if prediccion_k1[0] == 1 else "NO COMPRA (0)")

# Experimentación con K = 5
modelo_k5 = KNeighborsClassifier(n_neighbors=5)
modelo_k5.fit(X_entrenamiento, Y_entrenamiento)
prediccion_k5 = modelo_k5.predict(nuevo_cliente)
print("Predicción con K=5:", "COMPRA (1)" if prediccion_k5[0] == 1 else "NO COMPRA (0)")