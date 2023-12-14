import os
from carrega_histogramas import carrega_hist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

histogramas = []
rotulos = []

dataset_dir = r"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset"
for pasta in os.listdir(dataset_dir):
    if pasta != "histograms":
        dir_pasta = f"{dataset_dir}/{pasta}"
        for classe in os.listdir(dir_pasta):
            dir_classe = f"{dataset_dir}/{pasta}/{classe}"
            for imagem in os.listdir(dir_classe):
                imagem_path = f"{dataset_dir}/{pasta}/{classe}/{imagem}"
                #print(imagem_path)
                hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = carrega_hist(imagem_path)

                hist_phi_combined = np.concatenate([hist_phi_vivos, hist_phi_mortos])
                hist_psi_combined = np.concatenate([hist_psi_vivos, hist_psi_mortos])

                histograma_combinado = np.concatenate([hist_phi_combined, hist_psi_combined])

                histogramas.append(histograma_combinado)
                rotulos.append(classe)

X = np.array(histogramas)
y = np.array(rotulos)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# n estimators = 105
modelo = RandomForestClassifier(n_estimators=105, random_state=42)

modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)

precisao = accuracy_score(y_test, previsoes)

print("Precisão do modelo:", precisao)

for i, (previsao, verdadeiro) in enumerate(zip(previsoes, y_test)):
    resultado = "Correto" if previsao == verdadeiro else "Incorreto"
    print(f"Exemplo {i + 1}: Previsão={previsao}, Rótulo Verdadeiro={verdadeiro}, Resultado={resultado}")

