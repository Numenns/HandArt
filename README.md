# HandArt

Es una app de escritorio para dibujar mediante gestos de una mano, fue hecha en con una CNN de PyTorch previamente entrenada.

## Librerias usadas

- mediapipe 0.10.32
- opencv 4.14.0.94
- PySide6 6.10.2
- numpy 2.5.5
- torch 2.7.1

[MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python) · [Qt para Python](https://doc.qt.io/qtforpython-6/) · [PyTorch en Windows](https://pytorch.org/get-started/locally/) · [Documentación de OpenCV](https://docs.opencv.org/4.x/) · [Documentación de NumPy](https://numpy.org/doc/)

ademas se hace uso de Python 3.12

## Instalación y ejecución

### Desde Python en Windows

Instala Python 3.12 de 64 bits. Descarga o clona el proyecto y abre PowerShell en su carpeta raiz, donde se encuentran `handart/` y `assets/`.

Crea el entorno virtual.

```powershell
py -3.12 -m venv .venv
```

Instalacion de las libreiras

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install mediapipe==0.10.32 opencv-contrib-python==4.14.0.94 PySide6-Essentials==6.10.2 numpy==2.5.5
```

Ejecutar HandArt

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m handart
```

Después de instalar las dependencias, también puedes iniciar la app abriendo un  `scripts/ejecutar.bat`, que es con el que haciamos las pruebas, pero tambien contamos con un `HandArt.exe`, ábrelo con doble clic. El ejecutable generado se encuentra en `dist/HandArt.exe` e incluye Python, las librerias, los modelos y los iconos.


## Estructura de la aplicacion

| Archivo | Descripcion |
| :--- | :--- |
| `window.py` | Construye la interfaz grafica con PySide6 y traduce los gestos detectados en acciones dentro de la aplicacion. |
| `camera.py` | Captura el flujo de video con OpenCV y ejecuta el procesamiento de vision artificial en un hilo independiente. |
| `tracking.py` | Utiliza MediaPipe para detectar y extraer los 21 puntos clave de la mano. |
| `model.py` | Carga la red neuronal convolucional (misma CNN del parcial 2) entrenada para clasificar la entrada en uno de los cuatro gestos disponibles. |
| `gestures.py` | Combina las predicciones del modelo con la geometria de la mano (uso de mediapipe) y aplica filtrado para estabilizar el gesto activo. |
| `canvas.py` | Gestiona el lienzo interactivo controla el cursor, traza, borra, deshace cambios y exporta las imagenes. |
| `gesture_sidebar.py` | Renderiza los iconos SVG en la barra lateral y resalta visualmente el gesto que se encuentra activo. |

## Resultados

en [assets/modelos/regularizada_mejor.pt](assets/modelos/regularizada_mejor.pt) se pueden ver los siguientes datos

| Dato | Valor |
| :--- | ---: |
| Época guardada | 23 |
| Pérdida de validación (`val_loss`) | 0,104209 |
| Tamaño de entrada | 128 × 128 |
| Clases | 4 |
| Dropout | 0,3 |

El notebook de entrenamiento no está incluido en este repositorio, pero es el siguiente 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w1vaWzPSp3OYJQlCyXOuiyWnYY-6uAQI?usp=sharing)

### La matriz de confusion

| Real / Predicción | 2 dedos | Indice | Palma | Pinza |
| :--- | ---: | ---: | ---: | ---: |
| 2 dedos | 15 | 0 | 0 | 0 |
| Indice | 0 | 15 | 0 | 0 |
| Palma | 0 | 0 | 14 | 1 |
| Pinza | 0 | 0 | 0 | 15 |

### Rendimiento en tiempo real

[handart/camera.py](handart/camera.py) solicita una captura de 640 × 480 píxeles y ejecuta la CNN cuando MediaPipe detecta una mano.

[handart/window.py](handart/window.py) consulta el resultado más reciente mediante un temporizador configurado cada 16 ms

## Gestos

| Gesto | Accion |
| :--- | :--- |
| Indice | Dibuja trazos sobre el lienzo siguiendo la posicion del indice. |
| Palma | Activa el borrador para eliminar los trazos. |
| 2 dedos | Cambia al siguiente color disponible en la paleta. |
| Pinza | Ajusta el grosor (el cual es compartido entre el pincel y el borrador). |

## Video explicativo uso de la App

Demostración del funcionamiento de la App

[![HandArt](https://img.youtube.com/vi/7udjrg_LGgE/maxresdefault.jpg)](https://youtu.be/7udjrg_LGgE)
