import cv2
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