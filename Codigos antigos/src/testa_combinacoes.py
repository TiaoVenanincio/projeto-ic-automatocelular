# O objetivo desse código é testar diferentes combinações e "n_estimators"
#de histogramas para observar qual gera melhor acurácia para o modelo


import os
from src.carrega_histogramas import carrega_hist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def salva_info(index, combinacao, maior_precisao, estimators):
    #O objetivo dessa funcao é salvar as informações do melhor caso de treinamento para cada combinacao
    dir_txt = ('data/informacoes.txt')
    if index == 0:
        with open(dir_txt, 'w') as arquivo:
            #(f"index:combinacao:maior_precisao:n_estimators\n")
            arquivo.write(f"{index}:{combinacao}:{maior_precisao}:{estimators}\n")
    else:
        with open(dir_txt, 'a') as arquivo:
            arquivo.write(f"{index}:{combinacao}:{maior_precisao}:{estimators}\n")


def testador(images_dir):
    nome_combinacoes = ["hist_phi_vivos", "hist_phi_mortos", "hist_psi_vivos", "hist_psi_mortos",
                            "hist_phi_combined", "hist_psi_combined", "hist_vivos_combined", "hist_mortos_combined",
                            "all_combined"]
    for i in range(9):
        print("Testando combinacao: ", nome_combinacoes[i])
        histogramas = []
        rotulos = []
        for classe in os.listdir(images_dir):
            dir_classe = f"{images_dir}/{classe}"
            for imagem in os.listdir(dir_classe):
                imagem_path = f"{images_dir}/{classe}/{imagem}"

                hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = carrega_hist(imagem_path)

                #Cria diferentes combinações dos histogramas para avaliar qual se sai melhor
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        maior_precisao = 0
        #Aqui é testado um intervalo de valores para o classificador, tentando buscar um que trás a melhor acurácia
        for j in range(100, 251, 25):
            modelo = RandomForestClassifier(n_estimators=j, random_state=42)

            modelo.fit(X_train, y_train)

            previsoes = modelo.predict(X_test)

            precisao = accuracy_score(y_test, previsoes)
            print(f"N estimators: {j}, precisao do modelo: {precisao}")

            if precisao > maior_precisao: 
                maior_precisao = precisao
                estimators = j

        
        salva_info(i, nome_combinacoes[i], maior_precisao, estimators)
        print("")


        



