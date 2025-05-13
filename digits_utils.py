import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List

def add_noise_to_digits(x_values: List[List[float]], noise_level: float = 0.1) -> List[List[float]]:
    """
    Agrega ruido binario a un conjunto de imágenes de dígitos.
    Cambia un porcentaje de bits por imagen, sin alterar todos al azar.
    
    Args:
        x_values: Lista de imágenes, cada una como un vector de 35 valores binarios.
        noise_level: Proporción de bits a alterar por imagen (por ejemplo, 0.1 para 10%).

    Returns:
        Una nueva lista de imágenes con ruido.
    """
    noisy_images = []
    for image in x_values:
        noisy_image = image.copy()
        indices = list(range(len(image)))

        # En vez de elegir bits totalmente aleatorios, vamos a priorizar 1s (partes "activas" del dígito)
        one_indices = [i for i, val in enumerate(image) if val == 1]
        zero_indices = [i for i, val in enumerate(image) if val == 0]

        num_bits_to_flip = int(noise_level * len(image))
        flip_ones = int(0.6 * num_bits_to_flip)
        flip_zeros = num_bits_to_flip - flip_ones

        # Cambiar algunos 1s a 0
        if len(one_indices) > 0:
            to_flip_ones = random.sample(one_indices, min(flip_ones, len(one_indices)))
            for idx in to_flip_ones:
                noisy_image[idx] = 0

        # Cambiar algunos 0s a 1
        if len(zero_indices) > 0:
            to_flip_zeros = random.sample(zero_indices, min(flip_zeros, len(zero_indices)))
            for idx in to_flip_zeros:
                noisy_image[idx] = 1

        noisy_images.append(noisy_image)

    return noisy_images


def plot_digit_images(images: List[List[float]], cols: int = 10, figsize=(14, 8), title: str = "Dígitos"):
    """
    Muestra una grilla de imágenes binarias de dígitos (7x5), con fondo negro y píxeles blancos.
    
    Args:
        images: Lista de imágenes, cada una como un vector de 35 valores binarios.
        cols: Cantidad de columnas en la grilla.
        figsize: Tamaño total de la figura.
        title: Título del gráfico.
    """
    rows = (len(images) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize, facecolor='black')
    fig.suptitle(title, fontsize=18, color='white')

    # Asegurar que axs es siempre una matriz 2D
    axs = np.array(axs).reshape(rows, cols)

    for idx, image in enumerate(images):
        ax = axs[idx // cols, idx % cols]
        digit_array = np.array(image).reshape((7, 5))
        ax.imshow(digit_array, cmap="gray", interpolation="nearest")
        ax.axis("off")
        ax.set_facecolor("black")

    # Ocultar ejes vacíos si no se llenó la grilla
    for i in range(len(images), rows * cols):
        ax = axs[i // cols, i % cols]
        ax.axis("off")
        ax.set_facecolor("black")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.05, wspace=0.4, hspace=0.4)
    plt.show()