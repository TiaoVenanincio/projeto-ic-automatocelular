import os
from carrega_histogramas import carrega_hist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

dataset_dir = r"C:\Users\Sebastiao\Desktop\Projetos\projeto-ic-automatocelular\data\dataset"

# Listas para armazenar os histogramas e rótulos
histogramas = []
rotulos = []

for pasta in os.listdir(dataset_dir):
    if pasta != "histograms":
        dir_pasta = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset/{pasta}"
        for classe in os.listdir(dir_pasta):
            dir_classe = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset/{pasta}/{classe}"
            for imagem in os.listdir(dir_classe):
                imagem_path = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset/{pasta}/{classe}/{imagem}"
                #print(imagem_path)
                hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = carrega_hist(imagem_path)

                # Combine os histogramas conforme necessário
                hist_phi_combined = np.concatenate([hist_phi_vivos, hist_phi_mortos])
                hist_psi_combined = np.concatenate([hist_psi_vivos, hist_psi_mortos])

                # Especifique como você quer combinar os histogramas, isso é apenas um exemplo
                histograma_combinado = np.concatenate([hist_phi_combined, hist_psi_combined])

                # Adicione os histogramas combinados à lista
                histogramas.append(histograma_combinado)

                # Adicione o rótulo correspondente (classe) à lista
                rotulos.append(classe)

# Converta as listas para arrays numpy
X = np.array(histogramas)
y = np.array(rotulos)

# Divida os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie o modelo RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=150, random_state=42)

# Treine o modelo no conjunto de treino
modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)

precisao = accuracy_score(y_test, previsoes)

print("Precisão do modelo:", precisao)

# Mostrar resultados dos testes e se a previsão está correta
#for i, (previsao, verdadeiro) in enumerate(zip(previsoes, y_test)):
#    resultado = "Correto" if previsao == verdadeiro else "Incorreto"
#    print(f"Exemplo {i + 1}: Previsão={previsao}, Rótulo Verdadeiro={verdadeiro}, Resultado={resultado}")

