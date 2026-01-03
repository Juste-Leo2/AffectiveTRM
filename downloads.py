# downloads.py
import os
import urllib.request
import zipfile
import shutil
import sys
import ssl

# --- CONFIGURATION ---
URLS = {
    # Archive ZIP GitLab
    "dataset_zip": "https://gitlab.com/cs-cooper-lab/crema-d-mirror/-/archive/master/crema-d-mirror-master.zip",
    
    "llama_zip": "https://github.com/ggml-org/llama.cpp/releases/download/b7620/llama-b7620-bin-win-cpu-x64.zip",
    "gemma_model": "https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF/resolve/main/embeddinggemma-300M-qat-Q4_0.gguf",
    "enet_model": "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/enet_b0_8_best_afew.pt"
}

DIRS = {
    "dataset": "Data_collected",
    "llama": "llama_cpp",
    "weights": "weights"
}

def get_ssl_context():
    """Crée un contexte SSL permissif."""
    if hasattr(ssl, '_create_unverified_context'):
        return ssl._create_unverified_context()
    return None

def download_file(url, dest_path):
    print(f"   >> Téléchargement de {url} ...")
    try:
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=get_ssl_context()))
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 8192
            count = 0
            while True:
                buffer = response.read(block_size)
                if not buffer: break
                out_file.write(buffer)
                count += len(buffer)
                if total_size > 0:
                    percent = int(count * 100 / total_size)
                    sys.stdout.write(f"\r   >> Progression: {percent}% ({count//1024} KB)")
                    sys.stdout.flush()
        print("\n   >> Téléchargement terminé.")
        
    except Exception as e:
        print(f"\n   !! Erreur critique lors du téléchargement : {e}")
        # On nettoie le fichier corrompu si besoin
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)

def extract_zip(zip_path, extract_to, rename_root_to=None):
    print(f"   >> Extraction...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
        if rename_root_to:
            # Recherche du dossier extrait
            extracted_folders = [name for name in os.listdir(extract_to) 
                               if os.path.isdir(os.path.join(extract_to, name)) and name != "__MACOSX"]
            
            # On cherche le dossier qui semble correspondre à l'archive (souvent unique)
            target_extracted = None
            for f in extracted_folders:
                if "crema" in f.lower() or "mirror" in f.lower() or "master" in f.lower():
                    target_extracted = f
                    break
            
            if target_extracted:
                src_folder = os.path.join(extract_to, target_extracted)
                final_folder = os.path.join(extract_to, rename_root_to)
                
                if os.path.exists(final_folder):
                    print(f"   >> Le dossier {rename_root_to} existe déjà, fusion...")
                else:
                    os.rename(src_folder, final_folder)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== VÉRIFICATION DES DÉPENDANCES ET MODÈLES ===\n")

    # 1. DATASET
    target_data_dir = os.path.join(base_dir, DIRS["dataset"])
    print(f"[1/4] Vérification Dataset ({DIRS['dataset']})...")
    
    if os.path.exists(target_data_dir):
        print("   -> Dossier trouvé. Skip.")
    else:
        print("   -> Dossier introuvable. Installation...")
        zip_path = os.path.join(base_dir, "dataset_temp.zip")
        download_file(URLS["dataset_zip"], zip_path)
        extract_zip(zip_path, base_dir, rename_root_to=DIRS["dataset"])
        if os.path.exists(zip_path): os.remove(zip_path)
        print("   -> Dataset installé.")

    # 2. LLAMA.CPP
    target_llama_dir = os.path.join(base_dir, DIRS["llama"])
    print(f"\n[2/4] Vérification Llama.cpp ({DIRS['llama']})...")
    
    if os.path.exists(target_llama_dir):
        print("   -> Dossier trouvé. Skip.")
    else:
        print("   -> Dossier introuvable. Installation...")
        zip_path = os.path.join(base_dir, "llama_temp.zip")
        download_file(URLS["llama_zip"], zip_path)
        os.makedirs(target_llama_dir, exist_ok=True)
        extract_zip(zip_path, target_llama_dir)
        if os.path.exists(zip_path): os.remove(zip_path)
        print("   -> Llama.cpp installé.")

    # 3. DOSSIER WEIGHTS
    weights_dir = os.path.join(base_dir, DIRS["weights"])
    os.makedirs(weights_dir, exist_ok=True)

    # 4a. GEMMA MODEL
    gemma_path = os.path.join(weights_dir, "embeddinggemma-300M-qat-Q4_0.gguf")
    print(f"\n[3/4] Vérification Modèle Gemma...")
    
    if os.path.exists(gemma_path):
        print("   -> Fichier trouvé. Skip.")
    else:
        print("   -> Fichier introuvable. Téléchargement...")
        download_file(URLS["gemma_model"], gemma_path)

    # 4b. ENET MODEL
    enet_path = os.path.join(weights_dir, "enet_b0_8_best_afew.pt")
    print(f"\n[4/4] Vérification Modèle Enet...")
    
    if os.path.exists(enet_path):
        print("   -> Fichier trouvé. Skip.")
    else:
        print("   -> Fichier introuvable. Téléchargement...")
        download_file(URLS["enet_model"], enet_path)

    print("\n=== TOUT EST PRÊT. Vous pouvez lancer 'python prepare_data.py' ===")

if __name__ == "__main__":
    main()