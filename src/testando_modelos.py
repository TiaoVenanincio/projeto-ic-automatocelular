import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from carrega_histogramas import carrega_hist
from sklearn import preprocessing
from scipy.io.arff import loadarff
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
import time
import os

warnings.filterwarnings("ignore")

images_dir = f"data/dataset/images"

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

        #hist_vivos_combined = np.concatenate([hist_phi_vivos, hist_psi_vivos])
        #hist_mortos_combined = np.concatenate([hist_phi_mortos, hist_psi_mortos])

        all_combined = np.concatenate([hist_phi_combined, hist_psi_combined])

        #combinacoes = [hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos,
        #                    hist_phi_combined, hist_psi_combined, hist_vivos_combined, hist_mortos_combined,
        #                    all_combined]

        histogramas.append(all_combined)
        rotulos.append(classe)


X = np.array(histogramas)
y = np.array(rotulos)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Utilizando tecnicas de normalizacao
# 1 = MinMaxScaler, 2 = StandardScaler, 3 = MaxAbsScaler, 4 = RobustScaler
selectedNormalization = 1

if selectedNormalization == 1:
  scaler = preprocessing.MinMaxScaler()
if selectedNormalization == 2:
  scaler = preprocessing.StandardScaler()
if selectedNormalization == 3:
  scaler = preprocessing.MaxAbsScaler()
if selectedNormalization == 4:
  scaler = preprocessing.RobustScaler()
  
# Escalando os dados de treinamento
X_train = scaler.fit_transform(X_train)
# Escalando os dados de teste com os dados de treinamento, visto que os dados de teste podem ser apenas 1 amostra
X_test = scaler.transform(X_test)

print('Média do Conjunto de Treinamento por Feature:')
print(X_train.mean(axis = 0))
print('Desvio Padrão do Conjunto de Treinamento por Feature:')
print(X_train.std(axis = 0))

# Inicializar os classificadores

# Gaussian Naive Bayes
t = time.time()
gnb = GaussianNB()
model1 = gnb.fit(X_train, y_train)
print('Treino do Gaussian Naive Bayes Terminado. (Tempo de execucao: {})'.format(time.time() - t))

# Logistic Regression
t = time.time()
logreg = LogisticRegression()
model2 = logreg.fit(X_train, y_train)
print('Treino do Logistic Regression Terminado. (Tempo de execucao: {})'.format(time.time() - t))

# Decision Tree
t = time.time()
dectree = DecisionTreeClassifier()
model3 = dectree.fit(X_train, y_train)
print('Treino do Decision Tree Terminado. (Tempo de execucao: {})'.format(time.time() - t))

# K-Nearest Neighbors
t = time.time()
knn = KNeighborsClassifier(n_neighbors = 3)
model4 = knn.fit(X_train, y_train)
print('Treino do K-Nearest Neighbors Terminado. (Tempo de execucao: {})'.format(time.time() - t))
  
# Linear Discriminant Analysis
t = time.time()
lda = LinearDiscriminantAnalysis()
model5 = lda.fit(X_train, y_train)
print('Treino do Linear Discriminant Analysis Terminado. (Tempo de execucao: {})'.format(time.time() - t))

# Support Vector Machine
t = time.time()
svm = SVC()
model6 = svm.fit(X_train, y_train)
print('Treino do Support Vector Machine Terminado. (Tempo de execucao: {})'.format(time.time() - t))

# RandomForest
t = time.time()
rf = RandomForestClassifier()
model7 = rf.fit(X_train, y_train)
print('Treino do RandomForest Terminado. (Tempo de execucao: {})'.format(time.time() - t))

# Neural Net
t = time.time()
nnet = MLPClassifier(alpha=1)
model8 = nnet.fit(X_train, y_train)
print('Treino do Neural Net Terminado. (Tempo de execucao: {})'.format(time.time() - t))

# Cria 2 vetores de predicoes para armazenar todas acuracias e outros para as métricas
acc_train = []
acc_test = []
f1score = []
precision = []
recall = []

# Gaussian Naive Bayes

# Variavel para armazenar o tempo
t = time.time()
# Usando o modelo para predição das amostras de teste
aux = gnb.predict(X_test)
# Método para criar a matriz de confusão
cm = confusion_matrix(y_test, aux)
# Método para calcular o valor F1-Score
f1score.append(f1_score(y_test, aux, average = 'macro'))
# Método para calcular a Precision
precision.append(precision_score(y_test, aux, average = 'macro'))
# Método para calcular o Recall
recall.append(recall_score(y_test, aux, average = 'macro'))
# Salvando as acurácias nas listas
acc_train.append(gnb.score(X_train, y_train))
acc_test.append(gnb.score(X_test, y_test))
print('Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Treinamento: {:.2f}'.format(acc_train[0]))
print('Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Teste: {:.2f}'.format(acc_test[0]))
print('Matriz de Confusão:')
print(cm)
print('Precision: {:.5f}'.format(precision[0]))
print('Recall: {:.5f}'.format(recall[0]))
print('F1-score: {:.5f}'.format(f1score[0]))
print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
print('')

# Logistic Regression

# Variavel para armazenar o tempo
t = time.time()
# Usando o modelo para predição das amostras de teste
aux = logreg.predict(X_test)
# Método para criar a matriz de confusão
cm = confusion_matrix(y_test, aux)
# Método para calcular o valor F1-Score
f1score.append(f1_score(y_test, aux, average = 'macro'))
# Método para calcular a Precision
precision.append(precision_score(y_test, aux, average = 'macro'))
# Método para calcular o Recall
recall.append(recall_score(y_test, aux, average = 'macro'))
# Salvando as acurácias nas listas
acc_train.append(logreg.score(X_train, y_train))
acc_test.append(logreg.score(X_test, y_test))
print('Acuracia obtida com o Logistic Regression no Conjunto de Treinamento: {:.2f}'.format(acc_train[1]))
print('Acuracia obtida com o Logistic Regression no Conjunto de Teste: {:.2f}'.format(acc_test[1]))
print('Matriz de Confusão:')
print(cm)
print('Precision: {:.5f}'.format(precision[1]))
print('Recall: {:.5f}'.format(recall[1]))
print('F1-score: {:.5f}'.format(f1score[1]))
print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
print('')

# Decision Tree

# Variavel para armazenar o tempo
t = time.time()
# Usando o modelo para predição das amostras de teste
aux = dectree.predict(X_test)
# Método para criar a matriz de confusão
cm = confusion_matrix(y_test, aux)
# Método para calcular o valor F1-Score
f1score.append(f1_score(y_test, aux, average = 'macro'))
# Método para calcular a Precision
precision.append(precision_score(y_test, aux, average = 'macro'))
# Método para calcular o Recall
recall.append(recall_score(y_test, aux, average = 'macro'))
# Salvando as acurácias nas listas
acc_train.append(dectree.score(X_train, y_train))
acc_test.append(dectree.score(X_test, y_test))
print('Acuracia obtida com o Decision Tree no Conjunto de Treinamento: {:.2f}'.format(acc_train[2]))
print('Acuracia obtida com o Decision Tree no Conjunto de Teste: {:.2f}'.format(acc_test[2]))
print('Matriz de Confusão:')
print(cm)
print('Precision: {:.5f}'.format(precision[2]))
print('Recall: {:.5f}'.format(recall[2]))
print('F1-score: {:.5f}'.format(f1score[2]))
print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
print('')

# K-Nearest Neighbors

# Variavel para armazenar o tempo
t = time.time()
# Usando o modelo para predição das amostras de teste
aux = knn.predict(X_test)
# Método para criar a matriz de confusão
cm = confusion_matrix(y_test, aux)
# Método para calcular o valor F1-Score
f1score.append(f1_score(y_test, aux, average = 'macro'))
# Método para calcular a Precision
precision.append(precision_score(y_test, aux, average = 'macro'))
# Método para calcular o Recall
recall.append(recall_score(y_test, aux, average = 'macro'))
# Salvando as acurácias nas listas
acc_train.append(knn.score(X_train, y_train))
acc_test.append(knn.score(X_test, y_test))
print('Acuracia obtida com o K-Nearest Neighbors no Conjunto de Treinamento: {:.2f}'.format(acc_train[3]))
print('Acuracia obtida com o K-Nearest Neighbors no Conjunto de Teste: {:.2f}'.format(acc_test[3]))
print('Matriz de Confusão:')
print(cm)
print('Precision: {:.5f}'.format(precision[3]))
print('Recall: {:.5f}'.format(recall[3]))
print('F1-score: {:.5f}'.format(f1score[3]))
print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
print('')

# Linear Discriminant Analysis

# Variavel para armazenar o tempo
t = time.time()
# Usando o modelo para predição das amostras de teste
aux = lda.predict(X_test)
# Método para criar a matriz de confusão
cm = confusion_matrix(y_test, aux)
# Método para calcular o valor F1-Score
f1score.append(f1_score(y_test, aux, average = 'macro'))
# Método para calcular a Precision
precision.append(precision_score(y_test, aux, average = 'macro'))
# Método para calcular o Recall
recall.append(recall_score(y_test, aux, average = 'macro'))
# Salvando as acurácias nas listas
acc_train.append(lda.score(X_train, y_train))
acc_test.append(lda.score(X_test, y_test))
print('Acuracia obtida com o Linear Discriminant Analysis no Conjunto de Treinamento: {:.2f}'.format(acc_train[4]))
print('Acuracia obtida com o Linear Discriminant Analysis no Conjunto de Teste: {:.2f}'.format(acc_test[4]))
print('Matriz de Confusão:')
print(cm)
print('Precision: {:.5f}'.format(precision[4]))
print('Recall: {:.5f}'.format(recall[4]))
print('F1-score: {:.5f}'.format(f1score[4]))
print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
print('')

# Support Vector Machine

# Variavel para armazenar o tempo
t = time.time()
# Usando o modelo para predição das amostras de teste
aux = svm.predict(X_test)
# Método para criar a matriz de confusão
cm = confusion_matrix(y_test, aux)
# Método para calcular o valor F1-Score
f1score.append(f1_score(y_test, aux, average = 'macro'))
# Método para calcular a Precision
precision.append(precision_score(y_test, aux, average = 'macro'))
# Método para calcular o Recall
recall.append(recall_score(y_test, aux, average = 'macro'))
# Salvando as acurácias nas listas
acc_train.append(svm.score(X_train, y_train))
acc_test.append(svm.score(X_test, y_test))
print('Acuracia obtida com o Support Vector Machine no Conjunto de Treinamento: {:.2f}'.format(acc_train[5]))
print('Acuracia obtida com o Support Vector Machine no Conjunto de Teste: {:.2f}'.format(acc_test[5]))
print('Matriz de Confusão:')
print(cm)
print('Precision: {:.5f}'.format(precision[5]))
print('Recall: {:.5f}'.format(recall[5]))
print('F1-score: {:.5f}'.format(f1score[5]))
print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
print('')

# RandomForest

# Variavel para armazenar o tempo
t = time.time()
# Usando o modelo para predição das amostras de teste
aux = rf.predict(X_test)
# Método para criar a matriz de confusão
cm = confusion_matrix(y_test, aux)
# Método para calcular o valor F1-Score
f1score.append(f1_score(y_test, aux, average = 'macro'))
# Método para calcular a Precision
precision.append(precision_score(y_test, aux, average = 'macro'))
# Método para calcular o Recall
recall.append(recall_score(y_test, aux, average = 'macro'))
# Salvando as acurácias nas listas
acc_train.append(rf.score(X_train, y_train))
acc_test.append(rf.score(X_test, y_test))
print('Acuracia obtida com o RandomForest no Conjunto de Treinamento: {:.2f}'.format(acc_train[6]))
print('Acuracia obtida com o RandomForest no Conjunto de Teste: {:.2f}'.format(acc_test[6]))
print('Matriz de Confusão:')
print(cm)
print('Precision: {:.5f}'.format(precision[6]))
print('Recall: {:.5f}'.format(recall[6]))
print('F1-score: {:.5f}'.format(f1score[6]))
print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
print('')

# Neural Net

# Variavel para armazenar o tempo
t = time.time()
# Usando o modelo para predição das amostras de teste
aux = nnet.predict(X_test)
# Método para criar a matriz de confusão
cm = confusion_matrix(y_test, aux)
# Método para calcular o valor F1-Score
f1score.append(f1_score(y_test, aux, average = 'macro'))
# Método para calcular a Precision
precision.append(precision_score(y_test, aux, average = 'macro'))
# Método para calcular o Recall
recall.append(recall_score(y_test, aux, average = 'macro'))
# Salvando as acurácias nas listas
acc_train.append(nnet.score(X_train, y_train))
acc_test.append(nnet.score(X_test, y_test))
print('Acuracia obtida com o Neural Net no Conjunto de Treinamento: {:.2f}'.format(acc_train[7]))
print('Acuracia obtida com o Neural Net no Conjunto de Teste: {:.2f}'.format(acc_test[7]))
print('Matriz de Confusão:')
print(cm)
print('Precision: {:.5f}'.format(precision[7]))
print('Recall: {:.5f}'.format(recall[7]))
print('F1-score: {:.5f}'.format(f1score[7]))
print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
print('')