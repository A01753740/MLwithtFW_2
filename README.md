# Clasificación de Dígitos MNIST Usando CNN

Este proyecto implementa un modelo de Red Neuronal Convolucional (CNN) para clasificar dígitos escritos a mano (0-9) utilizando el conjunto de datos MNIST. El modelo se entrena con TensorFlow/Keras y logra una alta precisión tanto en la fase de entrenamiento como en la de validación.

## Resumen del Proyecto

El conjunto de datos MNIST contiene 70,000 imágenes en escala de grises de dígitos escritos a mano, cada una de 28x28 píxeles. El objetivo es clasificar cada imagen en una de las 10 clases de dígitos (0-9).

La arquitectura del modelo incluye capas convolucionales seguidas de capas de max-pooling y capas densas, utilizando **dropout** y regularización **L2** para prevenir el sobreajuste.

## Características Clave

- **División Train/Test/Validation**: Los datos se dividen en conjuntos de entrenamiento, validación y prueba para evaluar la generalización del modelo.
- **Análisis de Sesgo y Varianza**: Evaluados utilizando las métricas de precisión y pérdida durante el entrenamiento y la validación.
- **Regularización del Modelo**: Se aplican técnicas como **dropout** y regularización **L2** para prevenir el sobreajuste (*overfitting*).
- **Métricas de Evaluación**: Se calculan la **precisión**, el **recall** y el **F1-score** para medir el rendimiento del modelo en cada clase de dígitos.

## Requisitos

Para ejecutar el notebook, necesitarás instalar las siguientes dependencias:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

Puedes instalar estas bibliotecas utilizando `pip`:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```