import cv2
import numpy as np
import matplotlib.pyplot as plt

def amostrar_imagem(image, intervalo):    
    # Cria a imagem amostrada pegando pixels em intervalos regulares
    sampled_image = image[::intervalo, ::intervalo]
    
    return sampled_image

def exibir_imagens(imagens, intervalos):
    num_images = len(imagens)
    plt.figure(figsize=(15, 5))
    
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(cv2.cvtColor(imagens[i], cv2.COLOR_BGR2RGB))
        plt.title(f'Intervalo: {intervalos[i]}')
        plt.axis('off')
    
    plt.show()

def main():
    # Carrega a imagem
    caminho_imagem = 'img/bulb.jpg'
    imagem = cv2.imread(caminho_imagem)
    
    if imagem is None:
        print("Erro ao carregar a imagem.")
        return
    
    # Intervalos de amostragem diferentes
    intervalos = [1, 2, 4, 8, 16]
    
    # Lista para armazenar as imagens amostradas
    imagens_amostradas = [amostrar_imagem(imagem, intervalo) for intervalo in intervalos]
    
    # Exibe a imagem original e as amostradas
    exibir_imagens(imagens_amostradas, intervalos)

if __name__ == "__main__":
    main()
