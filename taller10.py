import numpy as np
from sklearn.svm import SVC

# ==========================================
# PUNTO 1: Dataset Original y Kernel Lineal
# ==========================================
# Se corrige 'x' minúscula por 'X' mayúscula que estaba en el código del PDF
X_original = np.array([
    [2, 2],
    [3, 3],
    [4, 2],
    [6, 6],
    [7, 8],
    [8, 7]
])
Y_original = np.array([0, 0, 0, 1, 1, 1])

# Entrenamiento con Kernel Lineal
modelo_lineal = SVC(kernel='linear')
modelo_lineal.fit(X_original, Y_original)

print("--- PUNTO 1: Modelo Lineal Original ---")
print("Vectores de Soporte descubiertos:\n", modelo_lineal.support_vectors_)

nuevo_punto = np.array([[5, 4]])
pred = modelo_lineal.predict(nuevo_punto)
print("Predicción para [5,4]: Clase", pred[0])
print()

# ==========================================
# PUNTOS 2, 3 y 4: "Engañando" a la frontera con punto [5,5]
# ==========================================
# Agregamos el punto [5,5] asignado a la Clase A (etiqueta 0)
X_modificado = np.array([
    [2, 2],
    [3, 3],
    [4, 2],
    [6, 6],
    [7, 8],
    [8, 7],
    [5, 5]  # Punto nuevo entrelazado
])
Y_modificado = np.array([0, 0, 0, 1, 1, 1, 0])

# Probar con Kernel Lineal (Intentará trazar recta rígida)
modelo_lineal_mod = SVC(kernel='linear')
modelo_lineal_mod.fit(X_modificado, Y_modificado)

# Probar con Kernel RBF (Creará una frontera curva no lineal)
modelo_rbf = SVC(kernel='rbf')
modelo_rbf.fit(X_modificado, Y_modificado)

print("--- PUNTOS 2-4: Evaluación con Punto [5,5] agregado ---")
print("Predicción [5,5] con Kernel Lineal:", modelo_lineal_mod.predict([[5, 5]])[0])
print("Predicción [5,5] con Kernel RBF:", modelo_rbf.predict([[5, 5]])[0])