import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Dados 
width = 140
height = 100
new_size = (width, height)  
image1_path = 'img/liberdade.jpg'
image2_path = 'img/cristo.jpg'

add_value = random.randint(50,100)
subtract_value = random.randint(50,100)
multiply_value = random.randint(0,9)/10 + 2
divide_value = random.randint(0,9)/10 + 2

# Ler a imagem e redimensionar
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Verifique se as imagens foram lidas corretamente
if img1 is None or img2 is None:
    print(f"Erro ao ler a imagem")
else:
    img1 = cv2.resize(img1, new_size)
    img2 = cv2.resize(img2, new_size)

def process_and_display_images(img):
    img_add= np.zeros(img.shape, dtype=np.uint8)
    img_subtract= np.zeros(img.shape, dtype=np.uint8)
    img_multiply= np.zeros(img.shape, dtype=np.uint8)
    img_divide= np.zeros(img.shape, dtype=np.uint8)
    cv2.normalize(cv2.add(img, add_value), img_add, 0, 255, cv2.NORM_MINMAX)  
    cv2.normalize(cv2.subtract(img, subtract_value), img_subtract, 0, 255, cv2.NORM_MINMAX)  
    cv2.normalize(cv2.multiply(img, multiply_value), img_multiply, 0, 255, cv2.NORM_MINMAX)  
    cv2.normalize(cv2.divide(img, divide_value), img_divide, 0, 255, cv2.NORM_MINMAX)  

    images = [img_add, img_subtract, img_multiply, img_divide]
    titles = [f'Addition {add_value}', f'Subtraction {subtract_value}', f'Multiplication {multiply_value}', f'Division {divide_value}']

    plt.figure(figsize=(12, 6))  

    # Usar subplots para exibir as imagens lado a lado
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i], fontsize=12)  # Título com tamanho de fonte ajustado
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def merge_images():
    img_add= np.zeros(img1.shape, dtype=np.uint8)
    img_subtract= np.zeros(img1.shape, dtype=np.uint8)
    img_multiply= np.zeros(img1.shape, dtype=np.uint8)
    cv2.normalize(cv2.add(img1, img2), img_add, 0, 255, cv2.NORM_MINMAX)  
    cv2.normalize(cv2.subtract(img1, img2), img_subtract, 0, 255, cv2.NORM_MINMAX)  
    cv2.normalize(cv2.multiply(img1, img2), img_multiply, 0, 255, cv2.NORM_MINMAX)  

    images = [img_add, img_subtract, img_multiply]
    titles = ['Addition', 'Subtraction', 'Multiplication']

    plt.figure(figsize=(12, 6))  

    # Usar subplots para exibir as imagens lado a lado
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i], fontsize=12)  # Título com tamanho de fonte ajustado
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
process_and_display_images(img1)
process_and_display_images(img2)
merge_images()