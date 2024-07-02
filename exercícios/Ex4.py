import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Dados 
new_size = (100, 100)  #(largura, altura)
background = 0 # 0 preto, 1 branco
image_path = 'img/Planets.jpg'
tolerance = 30

# Função para verificar se um pixel está dentro da tolerância de branco e preto
def is_white(pixel):
    return all(255 - tolerance<= channel <= 255 for channel in pixel)

def is_black(pixel):
    return all(0 <= channel <= tolerance for channel in pixel)

# Ler a imagem e redimensionar
img = cv2.imread(image_path)
img = cv2.resize(img, new_size)

# Obter as dimensões da imagem
height, width, _ = img.shape

# Criar um array numpy de zeros com a mesma dimensão da imagem
array = np.zeros((height, width), dtype=int)

# Inicializa o índice para valores diferentes de zero
i = 1
# Primeira varredura para marcar os pixels
for y in range(height):
    for x in range(width):
        # Obter a cor do pixel
        pixel = img[y, x]
        # Verificar se o pixel está dentro da tolerância de branco
        if background:
            if not is_white(pixel):
                array[y, x] = i
                i += 1
        else:
            if not is_black(pixel):
                array[y, x] = i
                i += 1

# Ajuste dos valores baseado nas adjacências
att = 1
while att == 1:
    att = 0
    for y in range(height):
        for x in range(width):
            if array[y, x] != 0:
                if x > 0 and array[y, x - 1] != 0 and array[y, x - 1] < array[y, x]:
                    array[y, x] = array[y, x - 1]
                    att = 1
                if y > 0 and array[y - 1, x] != 0 and array[y - 1, x] < array[y, x]:
                    array[y, x] = array[y - 1, x]
                    att = 1
                if x < width - 1 and array[y, x + 1] != 0 and array[y, x + 1] < array[y, x]:
                    array[y, x] = array[y, x + 1]
                    att = 1
                if y < height - 1 and array[y + 1, x] != 0 and array[y + 1, x] < array[y, x]:
                    array[y, x] = array[y + 1, x]
                    att = 1

unique_values = np.unique(array)

# Contagem dos componentes conectados (excluindo o valor 0)
num_components = len(unique_values) - 1  # Exclui o valor 0
print(f"Number of connected components: {num_components}")

# Mapear cada valor a uma cor aleatória
colors = {val: (random.random(), random.random(), random.random()) for val in unique_values}

# Criar uma imagem colorida com base no array
color_image = np.zeros((height, width, 3))

for y in range(height):
    for x in range(width):
        if array[y, x] != 0:
            color_image[y, x] = colors[array[y, x]]

# Plotar o array colorido
plt.imshow(color_image)
plt.title(f'Imagem Processada com Cores Randomizadas. {num_components} componentes')
plt.show()
