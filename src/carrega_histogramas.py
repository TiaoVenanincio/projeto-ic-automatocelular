import joblib

def carrega_hist(imagem_path):
    #imagem_path = r"C:/Users/Sebastiao/Desktop/Projetos/projeto-ic-automatocelular/data/dataset/test_set/beaches/100.jpg"
    imagem_path = imagem_path.split("/")
    pasta, classe, imagem = imagem_path[8], imagem_path[9], imagem_path[10]
    imagem = imagem.split(".")[0]

    histogramas_dir = "data/dataset/histograms"
    hist_phi_vivos = joblib.load(f"{histogramas_dir}/{pasta}/{classe}/{imagem}_phi_vivos.pkl")
    hist_phi_mortos = joblib.load(f"{histogramas_dir}/{pasta}/{classe}/{imagem}_phi_mortos.pkl")
    hist_psi_vivos = joblib.load(f"{histogramas_dir}/{pasta}/{classe}/{imagem}_psi_vivos.pkl")
    hist_psi_mortos = joblib.load(f"{histogramas_dir}/{pasta}/{classe}/{imagem}_psi_mortos.pkl")

    return hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos

#hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = carrega_hist()
#print(hist_phi_mortos)
#print(hist_phi_mortos)
#print(hist_psi_vivos)
#print(hist_psi_mortos)