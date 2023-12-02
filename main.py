import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def obter_vizinhos(matriz_de_intensidade, linha, coluna):
    #Esse método de partição da matriz foi escolhido para manter a analise dos pixels vizinhos
    #dentro dos limites da matriz de intensidade, evitando assim valores negativos ou fora do shape.
    vizinhos = matriz_de_intensidade[max(0, linha-1):min(matriz_de_intensidade.shape[0], linha+2),
                             max(0, coluna-1):min(matriz_de_intensidade.shape[1], coluna+2)]

    #print("Pixel e seus vizinhos:")
    #print(vizinhos)

    lista_de_vizinhos = vizinhos.flatten().tolist()

    #Remove o pixel central da lista
    lista_de_vizinhos.remove(matriz_de_intensidade[linha, coluna]) 

    return lista_de_vizinhos

def aplica_regras(lista_de_vizinhos, intensidade_pixel_central, estado_pixel_central):
    #Conta o número de vizinhos vivos com a mesma intensidade que o pixel central
    vizinhos_vivos = lista_de_vizinhos.count(intensidade_pixel_central)

    #Regra 1: A célula viva com dois ou três vizinhos vivos sobrevive
    if estado_pixel_central == 1 and (vizinhos_vivos == 2 or vizinhos_vivos == 3):
        return 1
    
    #Regra 2: A célula viva com menos de dois vizinhos vivos morre (subpopulação)
    elif estado_pixel_central == 1 and vizinhos_vivos < 2:
        return 0
    
    #Regra 3: A célula viva com mais de três vizinhos vivos morre (superpopulação)
    elif estado_pixel_central == 1 and vizinhos_vivos > 3:
        return 0
    
    #Regra 4: A célula morta com exatamente três vizinhos vivos se torna viva (resurreição)
    elif estado_pixel_central == 0 and vizinhos_vivos == 3:
        return 1
    
    #Pra todos os outros casos, a célula permanece no mesmo estado
    else:
        return estado_pixel_central

def percorre_imagem_aplicando_regras(matriz_de_estados, matriz_de_intensidade):
    for linha in range(matriz_de_intensidade.shape[0]):
        for coluna in range(matriz_de_intensidade.shape[1]):
            #Obtem os vizinhos do pixel atual
            lista_de_vizinhos = obter_vizinhos(matriz_de_intensidade, linha, coluna)
            #Aplica as regras do jogo da vida no pixel atual
            matriz_de_estados[linha, coluna] = aplica_regras(lista_de_vizinhos, matriz_de_intensidade[linha, coluna], matriz_de_estados[linha, coluna])
 
    return matriz_de_estados

#Carrega a imagem pre processada em escala de cinza
imagem_cinza =  cv2.imread(".\data\imagens_cinza\R2016.jpg", cv2.IMREAD_GRAYSCALE)

#Transforma a imagem em uma matriz com a intensidade de cada pixel
matriz_de_intensidade = np.array(imagem_cinza)

#Cria duas matrizes com os estados iniciais, uma com todos vivos, outra com todos mortos
matriz_de_estados_phi = np.ones(matriz_de_intensidade.shape, dtype=int)
matriz_de_estados_psi = np.zeros(matriz_de_intensidade.shape, dtype=int)

#matriz_de_intensidade = np.array([[1,1,1],[0,1,2],[3,4,3]])
#matriz_de_estados =  np.zeros(matriz_de_intensidade.shape, dtype=int)
#print(matriz_de_intensidade.shape)
#print(matriz_de_intensidade)
#print(matriz_de_estados.shape)
#print(matriz_de_estados)

#tempo_inicio = time.time()

#Aplica as regras do jogo da vida em cada uma das matrizes de estados iniciais
matriz_de_estados_phi = percorre_imagem_aplicando_regras(matriz_de_estados_phi, matriz_de_intensidade)
matriz_de_estados_psi = percorre_imagem_aplicando_regras(matriz_de_estados_psi, matriz_de_intensidade)

#tempo_fim = time.time()
#print(tempo_fim - tempo_inicio)

#print(matriz_de_estados_phi)

#Phi -> estado inicial = vivo
phi_vivos = matriz_de_intensidade.flatten()[matriz_de_estados_phi.flatten() == 1] #se manteram vivos
phi_mortos = matriz_de_intensidade.flatten()[matriz_de_estados_phi.flatten() == 0] #morreram


#Psi -> estado inicial = morto
psi_vivos = matriz_de_intensidade.flatten()[matriz_de_estados_psi.flatten() == 1] #ressuscitaram 
psi_mortos = matriz_de_intensidade.flatten()[matriz_de_estados_psi.flatten() == 0] #se manteram mortos

#print(phi_vivos)

plt.figure()
plt.hist(phi_vivos, bins=256, range=(0, 256), alpha=0.5, label='φ - Vivos')
plt.title('Histograma φ - Vivos')
plt.xlabel('Intensidade do Pixel')
plt.ylabel('Frequência')
plt.legend(loc='upper right')

plt.figure()
plt.hist(phi_mortos, bins=256, range=(0, 256), alpha=0.5, label='φ - Mortos')
plt.title('Histograma φ - Mortos')
plt.xlabel('Intensidade do Pixel')
plt.ylabel('Frequência')
plt.legend(loc='upper right')

plt.figure()
plt.hist(psi_vivos, bins=256, range=(0, 256), alpha=0.5, label='ψ - Vivos')
plt.title('Histograma ψ - Vivos')
plt.xlabel('Intensidade do Pixel')
plt.ylabel('Frequência')
plt.legend(loc='upper right')

plt.figure()
plt.hist(psi_mortos, bins=256, range=(0, 256), alpha=0.5, label='ψ - Mortos')
plt.title('Histograma ψ - Mortos')
plt.xlabel('Intensidade do Pixel')
plt.ylabel('Frequência')
plt.legend(loc='upper right')

plt.show()