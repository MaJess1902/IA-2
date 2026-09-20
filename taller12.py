import numpy as np

# Función de Activación: Sigmoide (devuelve un valor entre 0 y 1)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# ==========================================================
# RETO DIMENSIONAL: Procesamiento en Lote (2 Clientes a la vez)
# ==========================================================
# Matriz X de 2 clientes x 3 características: [Edad, Ingresos, Deuda]
X = np.array([
    [0.5, 0.8, 0.2],  # Cliente 1
    [0.1, 0.9, 0.9]   # Cliente 2
])

# 2. CAPA OCULTA (4 Neuronas)
# Matriz W1 de (3 entradas x 4 neuronas ocultas)
W1 = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [-0.5, 0.6, 0.7, -0.8],
    [0.9, -0.1, 0.2, 0.3]
])
b1 = np.array([0.1, 0.2, 0.3, 0.4])  # 4 Sesgos

# PROCESO CAPA OCULTA (Multiplicación matricial)
Z1 = np.dot(X, W1) + b1
A1 = sigmoide(Z1)  # Salida de la capa oculta (Transformada a rango 0-1)

print("--- VALORES DE LA CAPA OCULTA (Z1 y A1) ---")
print("Z1 (Valores puros):\n", Z1)
print("A1 (Activados con Sigmoide):\n", A1)
print()

# 3. CAPA DE SALIDA (1 Neurona final)
# Matriz W2 de (4 entradas ocultas x 1 neurona final)
W2 = np.array([0.5, 0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# PROCESO CAPA FINAL
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

print("--- PREDICCIÓN FINAL ---")
print("Probabilidades de aprobación para ambos clientes:", np.round(Salida_Final, 4))