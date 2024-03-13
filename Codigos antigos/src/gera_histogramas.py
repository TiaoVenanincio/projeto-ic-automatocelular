import os
import cv2
import numpy as np
import joblib


def obter_vizinhos(matriz_de_intensidade, linha, coluna):
    # O objetivo dessa função é retornar uma lista contendo a intensidade dos 9 vizinhos do pixel observado

    #Esse método de partição da matriz foi escolhido para manter a analise dos pixels vizinhos
    #dentro dos limites da matriz de intensidade, evitando assim valores negativos ou fora do shape.
    vizinhos = matriz_de_intensidade[max(0, linha-1):min(matriz_de_intensidade.shape[0], linha+2),
                             max(0, coluna-1):min(matriz_de_intensidade.shape[1], coluna+2)]

    #Transforma a matriz particionada em lista e remove o pixel do centro
    lista_de_vizinhos = vizinhos.flatten().tolist()
    lista_de_vizinhos.remove(matriz_de_intensidade[linha, coluna]) 

    return lista_de_vizinhos

def aplica_regras(lista_de_vizinhos, intensidade_pixel_central, estado_pixel_central):
    #O objetivo dessa função é aplicar as regras do jogo da vida de Conway no pixel observado e retornar seu estado.
    # 1 = vivo, 0 = morto

    #Conta o número de vizinhos vivos com a mesma intensidade que o pixel central
    vizinhos_iguais = lista_de_vizinhos.count(intensidade_pixel_central)

    #Regra 1: A célula viva com dois ou três vizinhos vivos sobrevive
    if estado_pixel_central == 1 and (vizinhos_iguais == 2 or vizinhos_iguais == 3):
        return 1
    
    #Regra 2: A célula viva com menos de dois vizinhos vivos morre (subpopulação)
    elif estado_pixel_central == 1 and vizinhos_iguais < 2:
        return 0
    
    #Regra 3: A célula viva com mais de três vizinhos vivos morre (superpopulação)
    elif estado_pixel_central == 1 and vizinhos_iguais > 3:
        return 0
    
    #Regra 4: A célula morta com exatamente três vizinhos vivos se torna viva (resurreição)
    elif estado_pixel_central == 0 and vizinhos_iguais == 3:
        return 1
    
    #Pra todos os outros casos, a célula permanece no mesmo estado
    else:
        return estado_pixel_central

def percorre_imagem_aplicando_regras(matriz_de_estados, matriz_de_intensidade):
    # O objetivo dessa função é percorrer a imagem, chamar a função para obter os vizinhos e aplicar as regras

    linhas, colunas = matriz_de_intensidade.shape
    for linha in range(linhas):
        for coluna in range(colunas):
            #Obtem os vizinhos do pixel atual
            lista_de_vizinhos = obter_vizinhos(matriz_de_intensidade, linha, coluna)
            #Aplica as regras do jogo da vida no pixel atual (atualiza a matriz de estado inicial)
            matriz_de_estados[linha, coluna] = aplica_regras(lista_de_vizinhos, matriz_de_intensidade[linha, coluna], matriz_de_estados[linha, coluna])
 
    return matriz_de_estados

def gera_histogramas(imagem_cinza):
    #O objetivo dessa função é criar as matriz de intensidade e as de estado inicial para cada imagem
    #Após aplicar as regras cria os histogramas

    #Transforma a imagem em uma matriz de intensidade
    matriz_de_intensidade = np.array(imagem_cinza)
    
    #Cria as matrizes de estados iniciais
    matriz_de_estados_phi = np.ones(matriz_de_intensidade.shape, dtype=int) #todos vivos
    matriz_de_estados_psi = np.zeros(matriz_de_intensidade.shape, dtype=int) #todos mortos

    #Aplica as regras do jogo da vida e atualiza as matrizes de estado inicial
    matriz_de_estados_phi = percorre_imagem_aplicando_regras(matriz_de_estados_phi, matriz_de_intensidade)
    matriz_de_estados_psi = percorre_imagem_aplicando_regras(matriz_de_estados_psi, matriz_de_intensidade)

    #As matrizes são convertidas em listas
    #Phi -> estado inicial = vivo
    phi_vivos = matriz_de_intensidade.flatten()[matriz_de_estados_phi.flatten() == 1] #se manteram vivos
    phi_mortos = matriz_de_intensidade.flatten()[matriz_de_estados_phi.flatten() == 0] #morreram

    #Psi -> estado inicial = morto
    psi_vivos = matriz_de_intensidade.flatten()[matriz_de_estados_psi.flatten() == 1] #ressuscitaram 
    psi_mortos = matriz_de_intensidade.flatten()[matriz_de_estados_psi.flatten() == 0] #se manteram mortos

    #Cria os histogramas
    hist_phi_vivos, _ = np.histogram(phi_vivos, bins=256, range=(0, 256))
    hist_phi_mortos, _ = np.histogram(phi_mortos, bins=256, range=(0, 256))
    hist_psi_vivos, _ = np.histogram(psi_vivos, bins=256, range=(0, 256))
    hist_psi_mortos, _ = np.histogram(psi_mortos, bins=256, range=(0, 256))

    return hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos

def gerador_histogramas(dataset_dir):
    #Funcao principal: faz a chamada das funções acima e salva os histogramas

    images_dir = f"{dataset_dir}/images"
    histograms_dir = f"{dataset_dir}/histograms"

    # Cria as pastas para salvar os histogramas mantendo o padrão das classes do dataset
    os.makedirs(histograms_dir, exist_ok=True)

    classes = [conteudo_item for conteudo_item in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, conteudo_item))]
    for classe in classes:
        dir_pastas = os.path.join(histograms_dir, classe)
        os.makedirs(dir_pastas, exist_ok=True)

    i = 0
    #Esse loop pega a imagem, gera seus histogramas e salva com base no nome da classe e da imagem
    for classe in os.listdir(images_dir):
        dir_classe = f"{images_dir}/{classe}"

        for imagem in os.listdir(dir_classe):
            imagem_path = f"{images_dir}/{classe}/{imagem}"

            imagem_cinza = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
            
            imagem_cinza = cv2.resize(imagem_cinza, (128,128))
            #Dimensao original = 384 * 256 = 98304
            #Resize = 128 * 128 = 16384
            #Redução = (16384-98304)/98304*100 = 83,33% 

            hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = gera_histogramas(imagem_cinza)

            index = imagem.split(".")[0]
            file_path = os.path.join(f"{histograms_dir}/{classe}", f"{index}_phi_vivos.pkl")
            joblib.dump(hist_phi_vivos, file_path)
            file_path = os.path.join(f"{histograms_dir}/{classe}", f"{index}_phi_mortos.pkl")
            joblib.dump(hist_phi_mortos, file_path)
            file_path = os.path.join(f"{histograms_dir}/{classe}", f"{index}_psi_vivos.pkl")
            joblib.dump(hist_psi_vivos, file_path)
            file_path = os.path.join(f"{histograms_dir}/{classe}", f"{index}_psi_mortos.pkl")
            joblib.dump(hist_psi_mortos, file_path)
                    
            i += 1
            print("Progresso = %0.1f por cento" % (i / 1000 * 100))

    os.system('clear')
    print("Progresso concluído")