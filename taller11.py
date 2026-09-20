import numpy as np

# 1. Función de Activación (Escalón)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

# 2. Estructura de la Neurona (Perceptrón)
def perceptron(X, W, b):
    # Producto punto (Combinación lineal)
    Z = np.dot(X, W) + b
    # Activación
    salida = funcion_escalon(Z)
    return salida

# =========================================================
# SOLUCIÓN AL RETO: COMPUERTA LÓGICA OR
# =========================================================
# Asignamos pesos = [0.5, 0.5] y sesgo = -0.2
pesos_or = np.array([0.5, 0.5])
sesgo_or = -0.2

# Casos de prueba para verificar la tabla de verdad OR
casos_prueba = [
    np.array([0, 0]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])
]

print("--- RESULTADOS COMPUERTA OR ---")
for entrada in casos_prueba:
    resultado = perceptron(entrada, pesos_or, sesgo_or)
    print(f"Entrada: {entrada} --> Salida Obtenida: {resultado}")