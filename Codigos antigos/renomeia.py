import os

dataset_dir = "./data/dataset_1"
images_dir = f"{dataset_dir}/images"

#Esse loop pega a imagem, gera seus histogramas e salva com base no nome da classe e da imagem
for classe in os.listdir(images_dir):
    dir_classe = f"{images_dir}/{classe}"
    i = 0
    for imagem in os.listdir(dir_classe):
        imagem_path = f"{images_dir}/{classe}/{imagem}"

        antigo = imagem_path
        novo = f"{dir_classe}/{classe}_{i}.jpg"

        os.rename(antigo, novo)
        i += 1