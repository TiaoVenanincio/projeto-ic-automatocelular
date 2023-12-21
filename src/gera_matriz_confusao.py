import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from carrega_histogramas import carrega_hist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def matriz_confusao(y_test, previsoes,nome, index, acuracia):
    matriz = confusion_matrix(y_test, previsoes)

    print(f"Gerando matriz de confusão para o modelo: {nome}...")

    plt.figure(figsize=(8, 8))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Matriz de Confusão - {nome} - {acuracia}')
    plt.xlabel('Rótulos Previstos')
    plt.ylabel('Rótulos Verdadeiros')
    plt.savefig(f'data/matrizes/MC{index}_{nome}.png')

def extrair_linha_por_indice(arquivo, indice):
    with open(arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
        if 0 <= indice < len(linhas):
            linha_especifica = linhas[indice].strip()
            return linha_especifica
        else:
            return None
        
def gera_log(previsoes, y_test, nome, i):
    with open(f'data/logs/{i}_{nome}.txt', 'w') as arquivo:
        for i, (previsao, verdadeiro) in enumerate(zip(previsoes, y_test)):
            resultado = "Correto" if previsao == verdadeiro else "Incorreto"
            dado = (f"Exemplo {i + 1}: Previsao={previsao}: Rotulo Verdadeiro={verdadeiro}: Resultado={resultado}\n")
            arquivo.write(dado)

images_dir = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset/images"
dir_matrizes = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/matrizes"
os.makedirs(dir_matrizes, exist_ok=True)

dir_logs = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/logs"
os.makedirs(dir_logs, exist_ok=True)

for i in range(9):
    histogramas = []
    rotulos = []
    for classe in os.listdir(images_dir):
        dir_classe = f"{images_dir}/{classe}"
        for imagem in os.listdir(dir_classe):
            imagem_path = f"{images_dir}/{classe}/{imagem}"

            hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = carrega_hist(imagem_path)

            hist_phi_combined = np.concatenate([hist_phi_vivos, hist_phi_mortos])
            hist_psi_combined = np.concatenate([hist_psi_vivos, hist_psi_mortos])

            hist_vivos_combined = np.concatenate([hist_phi_vivos, hist_psi_vivos])
            hist_mortos_combined = np.concatenate([hist_phi_mortos, hist_psi_mortos])

            all_combined = np.concatenate([hist_phi_combined, hist_psi_combined])

            combinacoes = [hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos,
                        hist_phi_combined, hist_psi_combined, hist_vivos_combined, hist_mortos_combined,
                        all_combined]

            histogramas.append(combinacoes[i])
            rotulos.append(classe)


    X = np.array(histogramas)
    y = np.array(rotulos)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    arquivo = "data/informacoes.txt"
    linha = extrair_linha_por_indice(arquivo, i)
    print("J = ",linha)
    j = linha.split(":")[3] #n_estimators que gerou maior precisao para cada combinacao

    modelo = RandomForestClassifier(n_estimators=int(j), random_state=42)

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    precisao = accuracy_score(y_test, previsoes)
    nome_combinacoes = ["hist_phi_vivos", "hist_phi_mortos", "hist_psi_vivos", "hist_psi_mortos",
                        "hist_phi_combined", "hist_psi_combined", "hist_vivos_combined", "hist_mortos_combined",
                        "all_combined"]
    
    #matriz_confusao(y_test, previsoes,nome_combinacoes[i], i, precisao)
    gera_log(previsoes, y_test, nome_combinacoes[i],i)