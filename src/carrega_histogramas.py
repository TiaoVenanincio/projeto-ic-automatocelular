import joblib

#O objetivo dessa função é buscar os histogramas associados a uma determinada imagem.

def carrega_hist(imagem_path):
    #data/dataset/beaches/100.jpg"
    imagem_path = imagem_path.split("/")
    classe, imagem = imagem_path[3], imagem_path[4]
    imagem = imagem.split(".")[0]

    histogramas_dir = "data/dataset/histograms"
    try:
        hist_phi_vivos = joblib.load(f"{histogramas_dir}/{classe}/{imagem}_phi_vivos.pkl")
        hist_phi_mortos = joblib.load(f"{histogramas_dir}/{classe}/{imagem}_phi_mortos.pkl")
        hist_psi_vivos = joblib.load(f"{histogramas_dir}/{classe}/{imagem}_psi_vivos.pkl")
        hist_psi_mortos = joblib.load(f"{histogramas_dir}/{classe}/{imagem}_psi_mortos.pkl")

        return hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos
    
    except:
        print("Os histogramas não foram encontrados. Verifique os diretórios e se os histogramas foram criados corretamente.")
        

#hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = carrega_hist()
#print(hist_phi_mortos)
#print(hist_phi_mortos)
#print(hist_psi_vivos)
#print(hist_psi_mortos)