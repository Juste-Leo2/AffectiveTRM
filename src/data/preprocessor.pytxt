# src/data/preprocessor.py
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from src.config import Config

# --- IMPORTS ENCODEURS ---
from src.encoders.text import start_server, get_batch_embeddings
from src.encoders.audio import audio_encoder_init, get_audio_embedding
from src.encoders.video import initialize_encoder, get_video_embedding

# --- CONFIGURATION DES ZONES ---
AV_CENTERS = {
    "ANG": [-0.6, 0.7], "DIS": [-0.7, 0.3], "FEA": [-0.3, 0.8],
    "HAP": [0.7, 0.6],  "SAD": [-0.7, -0.5], "NEU": [0.0, 0.0]
}
EMOTION_TO_ID = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "FEA": 4, "DIS": 5}
INTENSITY_SCALERS = {"LO": 0.6, "MD": 1.0, "HI": 1.2, "XX": 1.0}
EMOTIONS = list(AV_CENTERS.keys())
SENTENCE_MAP = {"IEO": "It's eleven o'clock", "TIE": "That is exactly what happened", "IOM": "I'm on my way to the meeting", "IWW": "I wonder what this is about", "TAI": "The airplane is almost full", "MTI": "Maybe tomorrow it will be cold", "IWL": "I would like a new alarm clock", "ITH": "I think I have a doctor's appointment", "DFA": "Don't forget a jacket", "ITS": "I think I've seen this before", "TSI": "The surface is slick", "WSI": "We'll stop in a couple of minutes"}

def map_emotion_to_av_cloud(emotion_code, intensity_code):
    center = np.array(AV_CENTERS[emotion_code], dtype=np.float32)
    scaler = INTENSITY_SCALERS.get(intensity_code, 1.0)
    if intensity_code == "XX": scaler = np.random.uniform(0.7, 1.2)
    target = center * scaler
    noise_scale = 0.08 if emotion_code == "NEU" else 0.15
    target = target + np.random.normal(loc=0.0, scale=noise_scale, size=2)
    return torch.tensor(np.clip(target, -1.0, 1.0), dtype=torch.float32)

def parse_crema_filename(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split('_')
    if len(parts) < 4: return None, None, None, None, None
    try: actor_id = int(parts[0])
    except ValueError: return None, None, None, None, None

    sentence_code, emotion_code, intensity_code = parts[1], parts[2], parts[3]
    if sentence_code not in SENTENCE_MAP or emotion_code not in EMOTIONS: return None, None, None, None, None
    
    return SENTENCE_MAP[sentence_code], map_emotion_to_av_cloud(emotion_code, intensity_code), EMOTION_TO_ID[emotion_code], base_name, actor_id

def preprocess_dataset():
    print(f"--- Pré-traitement Windows (Fix Dimensions) ---")
    raw_path = Config.DATA_RAW_DIR

    # 1. SCAN
    print("\n[ÉTAPE 1/5] Scan des fichiers...")
    all_audio_files = sorted(glob.glob(os.path.join(raw_path, "AudioWAV", "*.wav")))
    master_file_list = []
    for audio_path in tqdm(all_audio_files, desc="Scan"):
        audio_path = os.path.normpath(audio_path)
        sentence, label_av, label_class, base_name, actor_id = parse_crema_filename(audio_path)
        if sentence and os.path.exists(os.path.join(raw_path, "VideoFlash", base_name + ".flv")):
            master_file_list.append({
                'audio_path': audio_path, 'video_path': os.path.join(raw_path, "VideoFlash", base_name + ".flv"), 
                'sentence': sentence, 'label_av': label_av, 'label_class': label_class, 'base_name': base_name, 'actor_id': actor_id
            })
    print(f"Fichiers valides : {len(master_file_list)}")

    # 2. VIDEO
    if os.path.exists(Config.CACHE_VIDEO):
        video_embeddings_map = torch.load(Config.CACHE_VIDEO)
    else:
        print("\n[ÉTAPE 2/5] Traitement Vidéo...")
        initialize_encoder(Config.DEVICE)
        video_embeddings_map = {}
        for item in tqdm(master_file_list):
            if item['base_name'] not in video_embeddings_map:
                video_embeddings_map[item['base_name']] = get_video_embedding(item['video_path']).cpu()
        torch.save(video_embeddings_map, Config.CACHE_VIDEO)

    # 3. AUDIO
    if os.path.exists(Config.CACHE_AUDIO):
        audio_embeddings_map = torch.load(Config.CACHE_AUDIO)
    else:
        print("\n[ÉTAPE 3/5] Traitement Audio...")
        audio_model = audio_encoder_init()
        audio_model.to(Config.DEVICE)
        audio_embeddings_map = {}
        for item in tqdm(master_file_list):
            if item['base_name'] not in audio_embeddings_map:
                audio_embeddings_map[item['base_name']] = get_audio_embedding(audio_model, item['audio_path'], Config.DEVICE).cpu()
        torch.save(audio_embeddings_map, Config.CACHE_AUDIO)

    # 4. TEXTE
    if os.path.exists(Config.CACHE_TEXT):
        text_embedding_map = torch.load(Config.CACHE_TEXT)
    else:
        print("\n[ÉTAPE 4/5] Traitement Texte...")
        start_server(Config.TEXT_ENCODER_CONFIG['executable'], Config.TEXT_ENCODER_CONFIG['model'], Config.TEXT_ENCODER_CONFIG['port'])
        unique_sentences = list(set(item['sentence'] for item in master_file_list))
        text_embedding_map = {}
        for i in range(0, len(unique_sentences), 16):
            batch = unique_sentences[i:i+16]
            embeddings_list = get_batch_embeddings(batch, Config.TEXT_ENCODER_CONFIG['port'])
            if embeddings_list:
                for sent, emb_list in zip(batch, embeddings_list):
                    text_embedding_map[sent] = torch.tensor(emb_list, dtype=torch.float32)
        torch.save(text_embedding_map, Config.CACHE_TEXT)

    # 5. ASSEMBLAGE FINAL (CORRECTION DIMENSIONS)
    print("\n[ÉTAPE 5/5] Assemblage et Fusion...")
    final_data = []
    
    for item in tqdm(master_file_list, desc="Merging"):
        bn = item['base_name']
        if bn not in audio_embeddings_map or bn not in video_embeddings_map: continue
        
        audio_embs = audio_embeddings_map[bn]      
        video_embs = video_embeddings_map[bn]      
        text_emb = text_embedding_map[item['sentence']] 

        # --- CORRECTION CRITIQUE DES DIMENSIONS ---
        
        # 1. Audio : Dasheng sort souvent (1, T, D) -> on veut (T, D)
        if audio_embs.dim() == 3:
            audio_embs = audio_embs.squeeze(0) 
        
        # 2. Vidéo : Normalement (T, D), mais on vérifie
        if video_embs.dim() == 3:
            video_embs = video_embs.squeeze(0)

        # Maintenant, audio_embs et video_embs sont tous les deux 2D : (Time, Features)

        # Alignement temporel
        min_len = min(len(audio_embs), len(video_embs))
        if min_len == 0: continue
        
        # 3. Fusion
        # audio_embs[:min_len] -> (min_len, 768)
        # video_embs[:min_len] -> (min_len, 1280)
        # text_emb.unsqueeze(0).repeat -> (min_len, Dim_Texte)
        
        sequence_embeddings = torch.cat([
            audio_embs[:min_len], 
            video_embs[:min_len], 
            text_emb.unsqueeze(0).repeat(min_len, 1)
        ], dim=1) # dim=1 est la dimension des features
        
        final_data.append({
            'sequence_embeddings': sequence_embeddings, 
            'label_av': item['label_av'],
            'label_class': torch.tensor(item['label_class'], dtype=torch.long),
            'actor_id': item['actor_id']
        })
    
    print(f"Sauvegarde du dataset final dans : {Config.DATA_PROCESSED_PATH}")
    torch.save(final_data, Config.DATA_PROCESSED_PATH)
    print("--- Terminé avec succès ---")

if __name__ == "__main__":
    preprocess_dataset()