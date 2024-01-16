from src.testa_combinacoes import *
from src.gera_histogramas import *
from src.gera_matriz_confusao import *

dir_data = f"data"
dir_dataset = f"{dir_data}/dataset"
dir_images = f"{dir_dataset}/images"


#Se for a primeira vez executando o código, primeiro gere os histogramas, depois teste as combinações e, por fim, gere a matriz e os logs

print("... Menu ...")
print("1. Gerar histogramas para cada imagem")
print("2. Testar combinações dos histogramas")
print("3. Gerar matriz de confusao e logs")


opcao = 1
while opcao >= 1 and opcao <= 3:
    opcao =  int(input("\nInsira uma opcao: "))

    if opcao == 1:
        gerador_histogramas(dir_dataset)
    elif opcao == 2:
        testador(dir_images)
    elif opcao == 3:
        matriz_logs(dir_data)