import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dados
width = 200
height = 100 
MD = width*height
new_size = (width, height)
image_path = 'img/arcanine.jpg'

# Carregar a imagem
img = cv2.imread(image_path)

# Redimensionar a imagem
img_resized = cv2.resize(img, new_size)

# Converter para escala de cinza se a imagem for colorida
if len(img_resized.shape) == 3:
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img_resized

# Calcular o histograma
histogram = np.zeros(256, dtype=int)

# Iterar sobre cada pixel e contar a frequência de cada valor
height, width = img_gray.shape
for y in range(height):
    for x in range(width):
        pixel_value = img_gray[y, x]
        histogram[pixel_value] += 1

# probabilidade
sum = np.zeros(256, dtype=float)
s = np.zeros(256, dtype=float)
hist_att = np.zeros(256, dtype=float)
soma = 0
for i in range(256):
    sum[i] = histogram[i]/MD
    soma += sum[i]
    s[i] = 255 * soma
    hist_att[i] = round(s[i])

# Iterar sobre cada pixel e atualizar o valor
height, width = img_gray.shape
for y in range(height):
    for x in range(width):
        pixel_value = img_gray[y, x]
        img_gray[y, x] = hist_att[pixel_value]


name = 'output/ex7_histograma.png'
# Plotar o histograma
plt.figure(figsize=(10, 6))
plt.plot(histogram, color='black')
plt.title('Histograma da Imagem em Escala de Cinza')
plt.xlabel('Valor do Pixel')
plt.ylabel('Número de Pixels')
plt.grid(True)
plt.savefig(name)  

name = 'output/ex7_func_transformacao.png'
# Plotar o histograma
plt.figure(figsize=(10, 6))
plt.plot(s, color='black')
plt.title('Histograma da Imagem em Escala de Cinza')
plt.xlabel('Valor do Pixel')
plt.ylabel('Número de Pixels')
plt.grid(True)
plt.savefig(name)  

name = 'output/ex7_histograma_normalizado.png'
# Plotar o histograma
plt.figure(figsize=(10, 6))
plt.plot(hist_att, color='black')
plt.title('Histograma da Imagem em Escala de Cinza')
plt.xlabel('Valor do Pixel')
plt.ylabel('Número de Pixels')
plt.grid(True)
plt.savefig(name)  

plt.figure(figsize=(10, 6))
plt.plot(img_gray, color='black')
plt.title('imagem')
plt.grid(False)
plt.savefig('a.jpg')  