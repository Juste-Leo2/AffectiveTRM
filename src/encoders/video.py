# src/encoders/video.py (Partie corrigée)
import torch
import torch.nn as nn # Nécessaire pour nn.Identity
import numpy as np
import cv2
import os
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from torchvision import transforms
from src.config import Config
from PIL import Image
import timm 

# --- CLASSE PERSONNALISÉE CORRIGÉE ---
class LocalHSEmotion(HSEmotionRecognizer):
    def __init__(self, model_path, device='cpu'):
        # On ne lance PAS le super().__init__ pour éviter le téléchargement
        self.device = device
        self.is_mtcnn = False
        
        print(f"Chargement forcé du modèle local (Fix Timm): {model_path}")
        
        # 1. On charge le fichier .pt complet
        # map_location='cpu' pour éviter de surcharger le GPU pendant la manip
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 2. On crée une architecture EfficientNet B0 propre avec la version actuelle de timm
        # num_classes=8 car le fichier de poids contient la tête de classification pour 8 émotions
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=8)
        
        # 3. On extrait le dictionnaire de poids (state_dict) de l'objet checkpoint
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint
            
        # 4. On charge les poids dans le nouveau modèle
        # strict=False permet d'ignorer les différences mineures de nommage entre versions de timm
        self.model.load_state_dict(state_dict, strict=False)
        
        # 5. ADAPTATION IMPORTANTE : Sortie 1280
        # Ton VideoEncoder attend des embeddings de taille 1280 (final_embeddings = torch.zeros((..., 1280)))
        # Mais le modèle hsemotion sort 8 scores par défaut.
        # On remplace la dernière couche (classifier) par une identité pour récupérer les features.
        self.model.classifier = nn.Identity()
        
        self.model.to(device)
        self.model.eval()
        
        # Transformations standards
        self.idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}
        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Le reste de ta classe VideoEncoder reste identique...
class VideoEncoder:
    # ... (Garde ton code VideoEncoder tel quel)
    def __init__(self, device=None):
        self.device = device if device else Config.DEVICE
        print(f"Initialisation Encodeur Vidéo (HSEmotion + MTCNN) sur {self.device}...")
        
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)
        
        if os.path.exists(Config.VIDEO_MODEL_PATH):
            self.feature_extractor = LocalHSEmotion(model_path=Config.VIDEO_MODEL_PATH, device=self.device)
        else:
            print(f"ERREUR : Modèle introuvable {Config.VIDEO_MODEL_PATH}")
            # Fallback
            self.feature_extractor = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=self.device)

        print("Encodeur Vidéo prêt.")
    
    # ... (Garde le reste de tes méthodes process_video, etc.)
    def process_video(self, video_path):
        # (Copie ton code process_video ici tel quel)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erreur lecture vidéo: {video_path}")
            return torch.zeros((Config.NUM_FRAMES, 1280))

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = Config.NUM_FRAMES
        
        indices = np.linspace(0, total_frames - 1, Config.NUM_FRAMES, dtype=int)
        
        current_frame = 0
        frame_idx = 0
        
        batch_pil_images = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx < len(indices) and current_frame == indices[frame_idx]:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                batch_pil_images.append(pil_img)
                frame_idx += 1
            
            current_frame += 1
            if frame_idx >= len(indices): break
            
        cap.release()
        
        if not batch_pil_images:
             return torch.zeros((Config.NUM_FRAMES, 1280))

        # --- MTCNN ---
        try:
            faces = self.mtcnn(batch_pil_images) 
        except Exception as e:
            return torch.zeros((Config.NUM_FRAMES, 1280))
            
        valid_faces_imgs = []
        valid_indices = []

        for i, face in enumerate(faces):
            if face is not None:
                # MTCNN -> Numpy RGB
                face_np = face.permute(1, 2, 0).cpu().numpy()
                face_np = (face_np - face_np.min()) / (face_np.max() - face_np.min()) * 255
                face_np = face_np.astype(np.uint8)
                
                valid_faces_imgs.append(face_np)
                valid_indices.append(i)
        
        if not valid_faces_imgs:
            return torch.zeros((Config.NUM_FRAMES, 1280))

        # --- HSEmotion ---
        try:
            features = self.feature_extractor.extract_multi_features(valid_faces_imgs)
        except Exception as e:
            print(f"Erreur HSEmotion: {e}")
            return torch.zeros((Config.NUM_FRAMES, 1280))
        
        # --- Reconstitution ---
        final_embeddings = torch.zeros((Config.NUM_FRAMES, 1280))
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        for idx_in_features, original_idx in enumerate(valid_indices):
            if idx_in_features < len(features_tensor):
                final_embeddings[original_idx] = features_tensor[idx_in_features]
        
        # Padding
        for t in range(1, Config.NUM_FRAMES):
            if torch.all(final_embeddings[t] == 0) and not torch.all(final_embeddings[t-1] == 0):
                final_embeddings[t] = final_embeddings[t-1]

        return final_embeddings

_encoder_instance = None

def initialize_encoder(device):
    global _encoder_instance
    _encoder_instance = VideoEncoder(device)

def get_video_embedding(video_path):
    global _encoder_instance
    if _encoder_instance is None:
        initialize_encoder(Config.DEVICE)
    return _encoder_instance.process_video(video_path)