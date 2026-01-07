# src/encoders/video.py
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from torchvision import transforms
from src.config import Config
from PIL import Image
import timm 

# --- CLASSE PERSONNALISÉE (Fix Timm) ---
class LocalHSEmotion(HSEmotionRecognizer):
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.is_mtcnn = False
        print(f"Chargement forcé du modèle local (Fix Timm): {model_path}")
        
        # Chargement poids
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Création architecture
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=8)
        
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict, strict=False)
        
        # Sortie 1280 (Features)
        self.model.classifier = nn.Identity()
        
        self.model.to(device)
        self.model.eval()
        
        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class VideoEncoder:
    def __init__(self, device=None):
        self.device = device if device else Config.DEVICE
        print(f"Initialisation Encodeur Vidéo sur {self.device}...")
        
        # MTCNN pour la détection faciale (utilisé dans prepare_data)
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)
        
        if os.path.exists(Config.VIDEO_MODEL_PATH):
            self.feature_extractor = LocalHSEmotion(model_path=Config.VIDEO_MODEL_PATH, device=self.device)
        else:
            print(f"ERREUR : Modèle introuvable {Config.VIDEO_MODEL_PATH}")
            self.feature_extractor = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=self.device)

        print("Encodeur Vidéo prêt.")
    
    def process_video(self, video_path):
        """Extrait les embeddings d'un fichier vidéo complet."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.zeros((Config.NUM_FRAMES, 1280))

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = Config.NUM_FRAMES
        
        # Sélection uniforme des frames
        indices = np.linspace(0, total_frames - 1, Config.NUM_FRAMES, dtype=int)
        
        current_frame = 0
        frame_idx = 0
        batch_pil_images = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx < len(indices) and current_frame == indices[frame_idx]:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch_pil_images.append(Image.fromarray(frame_rgb))
                frame_idx += 1
            
            current_frame += 1
            if frame_idx >= len(indices): break
            
        cap.release()
        
        if not batch_pil_images:
             return torch.zeros((Config.NUM_FRAMES, 1280))

        # 1. Détection MTCNN
        try:
            faces = self.mtcnn(batch_pil_images) 
        except Exception:
            return torch.zeros((Config.NUM_FRAMES, 1280))
            
        valid_faces_imgs = []
        valid_indices = []

        for i, face in enumerate(faces):
            if face is not None:
                # MTCNN renvoie un tenseur normalisé, on le remet en image numpy pour HSEmotion
                face_np = face.permute(1, 2, 0).cpu().numpy()
                face_np = (face_np - face_np.min()) / (face_np.max() - face_np.min()) * 255
                face_np = face_np.astype(np.uint8)
                valid_faces_imgs.append(face_np)
                valid_indices.append(i)
        
        if not valid_faces_imgs:
            return torch.zeros((Config.NUM_FRAMES, 1280))

        # 2. Extraction Features
        try:
            features = self.feature_extractor.extract_multi_features(valid_faces_imgs)
        except Exception as e:
            print(f"Erreur HSEmotion: {e}")
            return torch.zeros((Config.NUM_FRAMES, 1280))
        
        # 3. Alignement Temporel
        final_embeddings = torch.zeros((Config.NUM_FRAMES, 1280))
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        for idx_in_features, original_idx in enumerate(valid_indices):
            if idx_in_features < len(features_tensor):
                final_embeddings[original_idx] = features_tensor[idx_in_features]
        
        # Padding (bouche-trou)
        for t in range(1, Config.NUM_FRAMES):
            if torch.all(final_embeddings[t] == 0) and not torch.all(final_embeddings[t-1] == 0):
                final_embeddings[t] = final_embeddings[t-1]

        return final_embeddings

    def extract_features_from_faces(self, face_images_list):
        """
        Nouvelle méthode pour l'inférence temps réel.
        Prend une liste d'images de visages DÉJÀ recadrés (numpy arrays).
        """
        if not face_images_list:
            return None
        try:
            # extract_multi_features attend une liste de numpy arrays
            features = self.feature_extractor.extract_multi_features(face_images_list)
            return torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            print(f"Erreur extract: {e}")
            return None

# --- INSTANCE GLOBALE ---
_encoder_instance = None

def initialize_encoder(device):
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = VideoEncoder(device)
    return _encoder_instance.feature_extractor.model # Retourne le modèle brut si besoin

def get_video_embedding(video_path):
    global _encoder_instance
    if _encoder_instance is None:
        initialize_encoder(Config.DEVICE)
    return _encoder_instance.process_video(video_path)

# --- LA FONCTION MANQUANTE QUE TU CHERCHAIS ---
def get_video_embedding_from_frames(frames_list):
    """
    Fonction wrapper pour satisfaire l'import de run_inference.py.
    Traite une liste de frames (images brutes ou visages).
    """
    global _encoder_instance
    if _encoder_instance is None:
        initialize_encoder(Config.DEVICE)
    
    # Si on passe des images brutes, c'est lourd car MTCNN va tourner.
    # Si on passe des visages recadrés, on appelle directement l'extracteur.
    # Pour l'inférence temps réel, on suppose que c'est des visages si on passe par ici.
    return _encoder_instance.extract_features_from_faces(frames_list)