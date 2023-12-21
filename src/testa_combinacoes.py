import os
from carrega_histogramas import carrega_hist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def salva_info(index, combinacao, maior_precisao, estimators):
    if index == 0:
        with open('data/informacoes.txt', 'w') as arquivo:
            #(f"index:combinacao:maior_precisao:n_estimators\n")
            arquivo.write(f"{index}:{combinacao}:{maior_precisao}:{estimators}\n")
    else:
        with open('data/informacoes.txt', 'a') as arquivo:
            arquivo.write(f"{index}:{combinacao}:{maior_precisao}:{estimators}\n")

images_dir = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset/images"

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

    maior_precisao = 0
    for j in range(100, 251, 25):
        modelo = RandomForestClassifier(n_estimators=j, random_state=42)

        modelo.fit(X_train, y_train)

        previsoes = modelo.predict(X_test)

        precisao = accuracy_score(y_test, previsoes)
        print("N estimators: ",  j)
        print("Precisão do modelo:", precisao)

        if precisao > maior_precisao: 
            maior_precisao = precisao
            estimators = j

    nome_combinacoes = ["hist_phi_vivos", "hist_phi_mortos", "hist_psi_vivos", "hist_psi_mortos",
                        "hist_phi_combined", "hist_psi_combined", "hist_vivos_combined", "hist_mortos_combined",
                        "all_combined"]
    
    salva_info(i, nome_combinacoes[i], maior_precisao, estimators)


        



