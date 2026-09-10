import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Crear el píxel BGR completamente amarillo (Azul=0, Verde=255, Rojo=255)
pixel = np.array([0, 255, 255], dtype=np.float32)

# 2. Pesos ponderados para el orden BGR
pesos = np.array([0.114, 0.587, 0.299])

# Producto punto entre el píxel y los pesos
gris_calculado = np.dot(pixel, pesos)

# 3. Imprimir el resultado
print(f"El valor de intensidad en gris para amarillo puro es: {gris_calculado:.2f}")
# Resultado aproximado: 225.93 (es decir, un tono de gris muy claro/brillante)

# 4. Verificación con una imagen real descargada usando OpenCV
imagen = cv2.imread('/workspaces/IA-2/imagenPrueba/image.png')

if imagen is not None:
    # Conversión mediante función optimizada de OpenCV
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    print("Conversión realizada con éxito en OpenCV.")
    print("Dimensiones de imagen original:", imagen.shape)
    print("Dimensiones de imagen en gris:", img_gris.shape)
else:
    print("Por favor, asegúrate de colocar la ruta de una imagen válida.")

#Segundo ejercicio

# 1. Cargar la imagen en la carpeta
imagen = cv2.imread('/workspaces/IA-2/imagenPrueba/image.png')  # Asegúrate de usar el nombre exacto de tu archivo

if imagen is None:
    print("Error: No se encontró la imagen. Asegúrate de haberla subido a la carpeta.")
else:
    # 2. Configurar colores para los canales en orden BGR (Azul, Verde, Rojo)
    colores = ('b', 'g', 'r')
    etiquetas = ('Canal Azul', 'Canal Verde', 'Canal Rojo')
    
    plt.figure(figsize=(10, 5))
    
    # 3. Calcular y graficar el histograma de cada canal
    for i, col in enumerate(colores):
        hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=etiquetas[i])
        plt.xlim([0, 256])

    plt.title("Histograma Comparativo de Canales RGB")
    plt.xlabel("Valor del Píxel (0 - 255)")
    plt.ylabel("Frecuencia (Cantidad de Píxeles)")
    plt.legend()
    plt.grid(True)
    
    # EN LUGAR DE plt.show(), GUARDAMOS LA IMAGEN GENERADA:
    plt.savefig('histograma_resultado.png')
    print("¡Proceso completado! La gráfica se guardó como 'histograma_resultado.png' en tu explorador de archivos.")