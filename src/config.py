# src/config.py
import os
import torch

class Config:
    # --- Chemins ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    DATA_RAW_DIR = os.path.join(BASE_DIR, "Data_collected")
    DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "preprocessed_data_hsemotion.pt")
    
    # Chemins Cache
    CACHE_MASTER_LIST = os.path.join(BASE_DIR, "src", "data", "master_file_list.pt")
    CACHE_AUDIO = os.path.join(BASE_DIR, "src", "data", "sequential_audio_embeddings.pt")
    CACHE_TEXT = os.path.join(BASE_DIR, "src", "data", "text_embedding_map.pt")
    CACHE_VIDEO = os.path.join(BASE_DIR, "src", "data", "sequential_video_embeddings_hsemotion.pt")
    
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "emotion_model_sequential_av.pth")
    
    # --- Modèles ---
    # On pointe vers le fichier local qu'on vient de télécharger
    VIDEO_MODEL_PATH = os.path.join(BASE_DIR, "weights", "enet_b0_8_best_afew.pt")
    
    TEXT_ENCODER_CONFIG = {
        'port': 8083,
        'executable': os.path.join(BASE_DIR, "llama_cpp", "llama-server.exe"), 
        'model': os.path.join(BASE_DIR, "weights", "embeddinggemma-300M-qat-Q4_0.gguf")
    }

    # --- Dimensions ---
    INPUT_DIM = 2816 # 768 + 768 + 1280
    HIDDEN_SIZE = 512
    NUM_CLASSES = 2      
    L_LAYERS = 2
    EXPANSION = 4
    NUM_HEADS = 8
    
    # --- Entraînement ---
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Paramètres Vidéo ---
    NUM_FRAMES = 30
    FRAME_SIZE = 224