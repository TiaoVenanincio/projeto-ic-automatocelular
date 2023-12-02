import cv2

def converter_para_escala_de_cinza(caminho_da_imagem, caminho_de_saida):
    imagem = cv2.imread(caminho_da_imagem)
    imagem_escala_de_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(caminho_de_saida, imagem_escala_de_cinza)

caminho_da_imagem = '.\data\imagens\R2016.jpg'
caminho_de_saida = '.\data\imagens_cinza\R2016.jpg'

converter_para_escala_de_cinza(caminho_da_imagem, caminho_de_saida)
