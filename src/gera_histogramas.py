import os
import cv2
import numpy as np
import joblib

def cria_pastas_hist(dir, tipo):
    conteudo = os.listdir(dir)
    classes = [conteudo_item for conteudo_item in conteudo if os.path.isdir(os.path.join(dir, conteudo_item))]
    for classe in classes:
        dir_pastas = f'C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset/histograms/{tipo}/{classe}'
        os.makedirs(dir_pastas, exist_ok=True)

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
    linhas, colunas = matriz_de_intensidade.shape
    for linha in range(linhas):
        for coluna in range(colunas):
            #Obtem os vizinhos do pixel atual
            lista_de_vizinhos = obter_vizinhos(matriz_de_intensidade, linha, coluna)
            #Aplica as regras do jogo da vida no pixel atual
            matriz_de_estados[linha, coluna] = aplica_regras(lista_de_vizinhos, matriz_de_intensidade[linha, coluna], matriz_de_estados[linha, coluna])
 
    return matriz_de_estados

def gera_histogramas(imagem_cinza):
    #Transforma a imagem em uma matriz com a intensidade de cada pixel
    matriz_de_intensidade = np.array(imagem_cinza)
    
    #Cria duas matrizes com os estados iniciais, uma com todos vivos, outra com todos mortos
    matriz_de_estados_phi = np.ones(matriz_de_intensidade.shape, dtype=int)
    matriz_de_estados_psi = np.zeros(matriz_de_intensidade.shape, dtype=int)

    matriz_de_estados_phi = percorre_imagem_aplicando_regras(matriz_de_estados_phi, matriz_de_intensidade)
    matriz_de_estados_psi = percorre_imagem_aplicando_regras(matriz_de_estados_psi, matriz_de_intensidade)

    #Phi -> estado inicial = vivo
    phi_vivos = matriz_de_intensidade.flatten()[matriz_de_estados_phi.flatten() == 1] #se manteram vivos
    phi_mortos = matriz_de_intensidade.flatten()[matriz_de_estados_phi.flatten() == 0] #morreram

    #Psi -> estado inicial = morto
    psi_vivos = matriz_de_intensidade.flatten()[matriz_de_estados_psi.flatten() == 1] #ressuscitaram 
    psi_mortos = matriz_de_intensidade.flatten()[matriz_de_estados_psi.flatten() == 0] #se manteram mortos

    hist_phi_vivos, _ = np.histogram(phi_vivos, bins=256, range=(0, 256))
    hist_phi_mortos, _ = np.histogram(phi_mortos, bins=256, range=(0, 256))
    hist_psi_vivos, _ = np.histogram(psi_vivos, bins=256, range=(0, 256))
    hist_psi_mortos, _ = np.histogram(psi_mortos, bins=256, range=(0, 256))

    return hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos

dataset_dir = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset"

histograms_dir = f"{dataset_dir}/histograms"
os.makedirs(histograms_dir, exist_ok=True)

train_dir = f"{dataset_dir}/training_set"
test_dir = f"{dataset_dir}/test_set"

#Criando as pastas para salvar o histograma
cria_pastas_hist(test_dir, "test_set")
cria_pastas_hist(train_dir, "training_set")

i = 0
for pasta in os.listdir(dataset_dir):
    if pasta != "histograms":
        dir_pasta = f"{dataset_dir}/{pasta}"
        for classe in os.listdir(dir_pasta):
            dir_classe = f"{dataset_dir}/{pasta}/{classe}"
            for imagem in os.listdir(dir_classe):
                imagem_path = f"{dataset_dir}/{pasta}/{classe}/{imagem}"
                imagem_cinza = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
                hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = gera_histogramas(imagem_cinza)

                index = imagem.split(".")[0]
                file_path = os.path.join(f"{histograms_dir}/{pasta}/{classe}", f"{index}_phi_vivos.pkl")
                joblib.dump(hist_phi_vivos, file_path)
                file_path = os.path.join(f"{histograms_dir}/{pasta}/{classe}", f"{index}_phi_mortos.pkl")
                joblib.dump(hist_phi_mortos, file_path)
                file_path = os.path.join(f"{histograms_dir}/{pasta}/{classe}", f"{index}_psi_vivos.pkl")
                joblib.dump(hist_psi_vivos, file_path)
                file_path = os.path.join(f"{histograms_dir}/{pasta}/{classe}", f"{index}_psi_mortos.pkl")
                joblib.dump(hist_psi_mortos, file_path)
                
                i += 1
                print("Progresso = %0.1f por cento" % (i / 1000 * 100))
