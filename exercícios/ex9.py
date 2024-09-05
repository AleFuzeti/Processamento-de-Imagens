import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para aplicar a convolução manualmente
def manual_convolution(img, mask):
    mask = np.flipud(np.fliplr(mask))
    img_height, img_width = img.shape
    mask_height, mask_width = mask.shape
    output_img = np.zeros((img_height, img_width), dtype=np.float32)
    padded_img = np.pad(img, ((mask_height//2, mask_height//2), (mask_width//2, mask_width//2)), mode='constant')
    
    for y in range(img_height):
        for x in range(img_width):
            roi = padded_img[y:y+mask_height, x:x+mask_width]
            conv_value = np.sum(roi * mask)
            output_img[y, x] = conv_value
    
    output_img = np.clip(output_img, 0, 255)
    output_img = output_img.astype(np.uint8)
    
    return output_img

# Função para aplicar os filtros passa-alta
def apply_high_pass_filters(img):
    laplace_mask = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=np.float32)

    sobel_x_mask = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32)
    
    sobel_y_mask = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=np.float32)

    prewitt_x_mask = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]], dtype=np.float32)
    
    prewitt_y_mask = np.array([[-1, -1, -1],
                               [ 0,  0,  0],
                               [ 1,  1,  1]], dtype=np.float32)

    # Máscaras Roberts 2x2
    roberts_x_mask = np.array([[ 1,  0],
                               [ 0, -1]], dtype=np.float32)
    
    roberts_y_mask = np.array([[ 0,  1],
                               [-1,  0]], dtype=np.float32)

    laplace_img = manual_convolution(img, laplace_mask)
    sobel_x_img = manual_convolution(img, sobel_x_mask)
    sobel_y_img = manual_convolution(img, sobel_y_mask)
    prewitt_x_img = manual_convolution(img, prewitt_x_mask)
    prewitt_y_img = manual_convolution(img, prewitt_y_mask)
    roberts_x_img = manual_convolution(img, roberts_x_mask)
    roberts_y_img = manual_convolution(img, roberts_y_mask)

    sobel_combined = cv2.addWeighted(sobel_x_img, 0.5, sobel_y_img, 0.5, 0)
    prewitt_combined = cv2.addWeighted(prewitt_x_img, 0.5, prewitt_y_img, 0.5, 0)
    roberts_combined = cv2.addWeighted(roberts_x_img, 0.5, roberts_y_img, 0.5, 0)

    return laplace_img, sobel_combined, prewitt_combined, roberts_combined

# Função para aplicar o filtro High Boost
def high_boost_filter(img, mask, k=1.5):
    blurred_img = manual_convolution(img, mask)
    high_boost_img = cv2.addWeighted(img, k, blurred_img, -1, 0)
    return high_boost_img

# Carregar e processar a imagem
def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    laplace_img, sobel_img, prewitt_img, roberts_img = apply_high_pass_filters(img)
    
    # Usando o filtro de média para suavização na filtragem high boost
    mean_mask = np.ones((3, 3), dtype=np.float32) / 9
    high_boost_img = high_boost_filter(img, mean_mask)

    return img, laplace_img, sobel_img, prewitt_img, roberts_img, high_boost_img

# Função para plotar os resultados
def plot_results(img, laplace_img, sobel_img, prewitt_img, roberts_img, high_boost_img):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(laplace_img, cmap='gray')
    plt.title('Filtro Laplace')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(sobel_img, cmap='gray')
    plt.title('Filtro Sobel')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(prewitt_img, cmap='gray')
    plt.title('Filtro Prewitt')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(roberts_img, cmap='gray')
    plt.title('Filtro Roberts')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(high_boost_img, cmap='gray')
    plt.title('Filtro High Boost')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Exemplo de uso
img_path = 'img/Planets.jpg'
img, laplace_img, sobel_img, prewitt_img, roberts_img, high_boost_img = process_image(img_path)
plot_results(img, laplace_img, sobel_img, prewitt_img, roberts_img, high_boost_img)
