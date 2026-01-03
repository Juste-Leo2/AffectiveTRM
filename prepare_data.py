# prepare_data.py
import os
import sys
from src.config import Config
from src.data.preprocessor import preprocess_dataset

if __name__ == "__main__":
    if not os.path.isdir(Config.DATA_RAW_DIR):
        print(f"Erreur : Le dossier du dataset '{Config.DATA_RAW_DIR}' est introuvable.")
        sys.exit(1)
        
    print("Lancement du pipeline de pré-traitement des données...")
    preprocess_dataset()