import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para quantizar a imagem
def quantize_image(image, levels):

    # Calculate the quantization step size
    step_size = 256 // levels
    
    # Apply quantization
    quantized_image = (image // step_size) * step_size
    
    return quantized_image

# Caminho da imagem
image_path = 'img/bulb.jpg'  # Substitua pelo caminho da sua imagem

# Ler a imagem usando o OpenCV
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if image is None:
    raise ValueError(f"Erro ao carregar a imagem no caminho: {image_path}")

# Níveis de quantização a serem testados
quantization_levels = [256, 128, 64, 32, 16, 8, 4, 2]

# Plotar as imagens quantizadas
plt.figure(figsize=(15, 10))

for i, levels in enumerate(quantization_levels):
    quantized_image = quantize_image(image, levels)
    
    plt.subplot(2, 4, i+1)
    plt.imshow(quantized_image, cmap='gray')
    plt.title(f'{levels} Níveis')
    plt.axis('off')

plt.tight_layout()
plt.show()
