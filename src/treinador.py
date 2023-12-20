import os
from carrega_histogramas import carrega_hist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

histogramas = []
rotulos = []
dataset_dir = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset"
images_dir = f"{dataset_dir}/images"

for classe in os.listdir(images_dir):
    dir_classe = f"{images_dir}/{classe}"
    for imagem in os.listdir(dir_classe):
        imagem_path = f"{images_dir}/{classe}/{imagem}"
        #print(imagem_path)
        hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = carrega_hist(imagem_path)

        hist_phi_combined = np.concatenate([hist_phi_vivos, hist_phi_mortos])
        hist_psi_combined = np.concatenate([hist_psi_vivos, hist_psi_mortos])

        histograma_combinado = np.concatenate([hist_phi_combined, hist_psi_combined])

        histogramas.append(histograma_combinado)
        rotulos.append(classe)


X = np.array(histogramas)
y = np.array(rotulos)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# n estimators = 105
for i in range(100, 251, 10):
    modelo = RandomForestClassifier(n_estimators=i, random_state=42)

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    precisao = accuracy_score(y_test, previsoes)
    print("N estimators: ",  i)
    print("Precisão do modelo:", precisao)

#for i, (previsao, verdadeiro) in enumerate(zip(previsoes, y_test)):
#    resultado = "Correto" if previsao == verdadeiro else "Incorreto"
#    print(f"Exemplo {i + 1}: Previsão={previsao}, Rótulo Verdadeiro={verdadeiro}, Resultado={resultado}")

# Criar a matriz de confusão
matriz_confusao = confusion_matrix(y_test, previsoes)

# Visualizar a matriz de confusão com seaborn
plt.figure(figsize=(8, 8))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Matriz de Confusão')
plt.xlabel('Rótulos Previstos')
plt.ylabel('Rótulos Verdadeiros')
plt.show()