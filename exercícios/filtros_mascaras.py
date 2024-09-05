import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

def convolution(img, mask):
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
    mean_img = convolution(img, mask)
    thresholded_img = np.where(mean_img > threshold, 255, 0)
    return thresholded_img

# Função para aplicar o filtro da mediana
def median_filter(img, size=3):
    img_height, img_width = img.shape
    output_img = np.zeros((img_height, img_width), dtype=np.uint8)
    
    padded_img = np.pad(img, ((size//2, size//2), (size//2, size//2)), mode='constant')
    
    # Aplicar da mediana
    for y in range(img_height):
        for x in range(img_width):
            roi = padded_img[y:y+size, x:x+size]
            median_value = np.median(roi)
            output_img[y, x] = median_value
    
    return output_img

# Função para aplicar o filtro da moda
def mode_filter(img, size=3):
    img_height, img_width = img.shape
    output_img = np.zeros((img_height, img_width), dtype=np.uint8)
    
    padded_img = np.pad(img, ((size//2, size//2), (size//2, size//2)), mode='constant')
    
    for y in range(img_height):
        for x in range(img_width):
            roi = padded_img[y:y+size, x:x+size].flatten()
            
            # Contar a frequência de cada valor na ROI
            values, counts = np.unique(roi, return_counts=True)
            
            # Encontrar o valor com a maior frequência (moda)
            max_count_index = np.argmax(counts)
            mode_value = values[max_count_index]
            
            output_img[y, x] = mode_value
    
    return output_img

# Função para aplicar o filtro gaussiano
def gaussian_filter(img, size=3, sigma=1):
    img_height, img_width = img.shape
    output_img = np.zeros((img_height, img_width), dtype=np.float32)
    
    padded_img = np.pad(img, ((size//2, size//2), (size//2, size//2)), mode='constant')
    
    # Criar uma máscara gaussiana
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    gaussian_mask = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    gaussian_mask = gaussian_mask / np.sum(gaussian_mask)
    
    # Aplicar a convolução
    for y in range(img_height):
        for x in range(img_width):
            roi = padded_img[y:y+size, x:x+size]
            conv_value = np.sum(roi * gaussian_mask)
            output_img[y, x] = conv_value
    
    output_img = np.clip(output_img, 0, 255)
    output_img = output_img.astype(np.uint8)
    
    return output_img

# Função para aplicar o filtro de Laplace
def laplace_filter(img, mask):
    laplace_img = convolution(img, mask)
    return laplace_img

# Função para aplicar o filtro de High-Boost
def high_boost_filter(img, mask, k=1):
    high_boost_img = img + k * convolution(img, mask)
    high_boost_img = np.clip(high_boost_img, 0, 255)
    high_boost_img = high_boost_img.astype(np.uint8)
    return high_boost_img

# Função para aplicar o filtro de Roberts
def roberts_filter(img):
    # Horizontal
    roberts_mask1 = np.array([[1, 0], [0, -1]])
    # Vertical
    roberts_mask2 = np.array([[0, 1], [-1, 0]])
    
    roberts_img1 = convolution(img, roberts_mask1)
    roberts_img2 = convolution(img, roberts_mask2)
    
    roberts_img = np.sqrt(roberts_img1**2 + roberts_img2**2)
    # Normalizar os valores para estar no intervalo [0, 255]
    roberts_img = np.clip(roberts_img, 0, 255)
    roberts_img = roberts_img.astype(np.uint8)
    
    return roberts_img

# Função para aplicar o filtro de Sobel
def sobel_filter(img):
    # Horizontal
    sobel_mask1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # Vertical
    sobel_mask2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    sobel_img1 = convolution(img, sobel_mask1)
    sobel_img2 = convolution(img, sobel_mask2)
    
    sobel_img = np.sqrt(sobel_img1**2 + sobel_img2**2)
    # Normalizar
    sobel_img = np.clip(sobel_img, 0, 255)
    sobel_img = sobel_img.astype(np.uint8)
    
    return sobel_img

# Função para aplicar o filtro de Prewitt
def prewitt_filter(img):
    # Horizontal
    prewitt_mask1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # Vertical
    prewitt_mask2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    prewitt_img1 = convolution(img, prewitt_mask1)
    prewitt_img2 = convolution(img, prewitt_mask2)
    
    prewitt_img = np.sqrt(prewitt_img1**2 + prewitt_img2**2)
    # Normalizar 
    prewitt_img = np.clip(prewitt_img, 0, 255)
    prewitt_img = prewitt_img.astype(np.uint8)
    
    return prewitt_img

# Filtro de média (3x3)
mean_mask = np.ones((3, 3), dtype=np.float32) / 9

# Filtro de Laplace
laplace_mask = np.array([[0, -1, 0], 
                         [-1, 4, -1], 
                         [0, -1, 0]], dtype=np.float32)

# Filtro de High-Boost
high_boost_mask = np.array([[0, -1, 0], 
                            [-1, 5, -1], 
                            [0, -1, 0]], dtype=np.float32)

# Carregar e processar imagem
def process_image(img):
    # Aplicar filtros manualmente
    mean_img = convolution(img, mean_mask)
    thresholded_mean_img = thresholded_mean_filter(img, mean_mask)
    median_img = median_filter(img)
    mode_img = mode_filter(img)
    gaussian_img = gaussian_filter(img)
    
    # Aplicar filtros de passa-alta
    laplace_img = laplace_filter(img, laplace_mask)
    high_boost_img = high_boost_filter(img, high_boost_mask)
    roberts_img = roberts_filter(img)
    sobel_img = sobel_filter(img)
    prewitt_img = prewitt_filter(img)
    
    return mean_img, thresholded_mean_img, median_img, mode_img, gaussian_img, laplace_img, high_boost_img, roberts_img, sobel_img, prewitt_img

# Plotagem das imagens processadas
def plot_results(img, mean_img, thresholded_mean_img, median_img, mode_img, gaussian_img, laplace_img, high_boost_img, roberts_img, sobel_img, prewitt_img):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 5, 3)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(3, 5, 6)
    plt.imshow(mean_img, cmap='gray')
    plt.title('Filtro de Média')
    plt.axis('off')
    
    plt.subplot(3, 5, 7)
    plt.imshow(thresholded_mean_img, cmap='gray')
    plt.title('Filtro de Média com Limiar')
    plt.axis('off')
    
    plt.subplot(3, 5, 8)
    plt.imshow(median_img, cmap='gray')
    plt.title('Filtro de Mediana')
    plt.axis('off')
    
    plt.subplot(3, 5, 9)
    plt.imshow(mode_img, cmap='gray')
    plt.title('Filtro de Moda')
    plt.axis('off')
    
    plt.subplot(3, 5, 10)
    plt.imshow(gaussian_img, cmap='gray')
    plt.title('Filtro Gaussiano')
    plt.axis('off')
    
    plt.subplot(3, 5, 11)
    plt.imshow(laplace_img, cmap='gray')
    plt.title('Filtro de Laplace')
    plt.axis('off')
    
    plt.subplot(3, 5, 12)
    plt.imshow(high_boost_img, cmap='gray')
    plt.title('Filtro High-Boost')
    plt.axis('off')
    
    plt.subplot(3, 5, 13)
    plt.imshow(roberts_img, cmap='gray')
    plt.title('Filtro de Roberts')
    plt.axis('off')
    
    plt.subplot(3, 5, 14)
    plt.imshow(sobel_img, cmap='gray')
    plt.title('Filtro de Sobel')
    plt.axis('off')
    
    plt.subplot(3, 5, 15)
    plt.imshow(prewitt_img, cmap='gray')
    plt.title('Filtro de Prewitt')
    plt.axis('off')
    
    plt.show()
    
# Carregar e processar a imagem de exemplo
img_resize = cv2.imread('img/lenna.png', cv2.IMREAD_GRAYSCALE)
# Redimensionar a imagem
img = cv2.resize(img_resize, (300, 300))

# Processar a imagem já carregada
mean_img, thresholded_mean_img, median_img, mode_img, gaussian_img, laplace_img, high_boost_img, roberts_img, sobel_img, prewitt_img = process_image(img)

# Plotar os resultados
plot_results(img, mean_img, thresholded_mean_img, median_img, mode_img, gaussian_img, laplace_img, high_boost_img, roberts_img, sobel_img, prewitt_img)