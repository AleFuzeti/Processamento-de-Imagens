import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

def manual_convolution(img, mask):
    # Rotacionar a máscara em 180° para realizar a convolução
    mask = np.flipud(np.fliplr(mask))
    
    # Pegar as dimensões da imagem e da máscara
    img_height, img_width = img.shape
    mask_height, mask_width = mask.shape

    # Criar uma imagem de saída de mesmo tamanho, inicializada com zeros
    output_img = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Pad da imagem original para evitar problemas nas bordas
    padded_img = np.pad(img, ((mask_height//2, mask_height//2), (mask_width//2, mask_width//2)), mode='constant')
    
    # Aplicar a convolução
    for y in range(img_height):
        for x in range(img_width):
            # Extrair a região de interesse (ROI)
            roi = padded_img[y:y+mask_height, x:x+mask_width]
            
            # Multiplicar a máscara pela ROI e calcular a soma
            conv_value = np.sum(roi * mask)
            
            # Atribuir o valor resultante ao pixel correspondente na imagem de saída
            output_img[y, x] = conv_value
    
    # Normalizar os valores para estar no intervalo [0, 255]
    output_img = np.clip(output_img, 0, 255)
    output_img = output_img.astype(np.uint8)
    
    return output_img

# Função para aplicar o filtro da média com limiar
def thresholded_mean_filter(img, mask, threshold=128):
    mean_img = manual_convolution(img, mask)
    thresholded_img = np.where(mean_img > threshold, 255, 0)
    return thresholded_img

# Função para aplicar o filtro da mediana manualmente
def manual_median_filter(img, size=3):
    img_height, img_width = img.shape
    output_img = np.zeros((img_height, img_width), dtype=np.uint8)
    
    padded_img = np.pad(img, ((size//2, size//2), (size//2, size//2)), mode='constant')
    
    for y in range(img_height):
        for x in range(img_width):
            roi = padded_img[y:y+size, x:x+size]
            median_value = np.median(roi)
            output_img[y, x] = median_value
    
    return output_img

# Máscara de exemplo para filtro de média (3x3)
mean_mask = np.ones((3, 3), dtype=np.float32) / 9

# Carregar e processar imagem
def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Aplicar filtros manualmente
    mean_img = manual_convolution(img, mean_mask)
    thresholded_mean_img = thresholded_mean_filter(img, mean_mask)
    median_img = manual_median_filter(img)
    
    return img, mean_img, thresholded_mean_img, median_img

# Plotagem das imagens processadas
def plot_results(img, mean_img, thresholded_mean_img, median_img):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(mean_img, cmap='gray')
    plt.title('Filtro de Média')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(thresholded_mean_img, cmap='gray')
    plt.title('Filtro de Média com Limiar')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(median_img, cmap='gray')
    plt.title('Filtro de Mediana')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Exemplo de uso
img_path = 'img/arcanine.jpg'
img, mean_img, thresholded_mean_img, median_img = process_image(img_path)
plot_results(img, mean_img, thresholded_mean_img, median_img)