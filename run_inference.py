# run_inference.py
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
import threading
import queue
import time
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from torchvision import transforms
from PIL import Image

# Imports locaux
from src.config import Config
from src.models.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from src.encoders.audio import audio_encoder_init, get_audio_embedding
from src.encoders.video import initialize_encoder
from src.encoders.text import start_server, get_batch_embeddings

# --- PARAMÈTRES TEMPS RÉEL ---
PREDICTION_INTERVAL_S = 0.5  # Une prédiction toutes les 0.5s
SAMPLE_RATE = 16000          # Audio 16kHz requis par Wav2Vec/HuBERT
VIDEO_FPS = 30               # Webcam standard
BUFFER_DURATION = 1.0        # Fenêtre glissante pour l'analyse

class RealTimeEmotionTester:
    def __init__(self):
        self.device = Config.DEVICE
        print(f"\n📱 Démarrage Inference sur : {self.device}")
        
        # 1. Initialisation des composants
        self._init_transforms()
        self._load_encoders()
        self._start_text_server()
        self._load_model()
        self._init_hardware()
        
        # 2. État interne
        self.audio_queue = queue.Queue()
        self.video_buffer = []
        self.is_running = False
        self.last_pred_time = 0
        
        # Lissage des prédictions (moyenne mobile)
        self.val_pred = 0.0
        self.aro_pred = 0.0
        
        # Mémoire du modèle (Carry)
        self.trm_carry = None

    def _init_transforms(self):
        # Transformation identique à celle attendue par EfficientNet (ImageNet stats)
        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Normalisation globale des entrées (comme dans train.py)
        self.input_normalizer = nn.LayerNorm(Config.INPUT_DIM).to(self.device)

    def _load_encoders(self):
        print("🔧 Chargement des encodeurs...")
        # Audio
        self.audio_model = audio_encoder_init()
        self.audio_model.to(self.device)
        self.audio_model.eval()
        
        # Vidéo
        # Note: initialize_encoder charge le modèle et le met en globale ou le retourne
        # Ici on suppose qu'il retourne le modèle ou qu'on peut l'utiliser
        # Dans votre code original, initialize_encoder renvoyait un objet ou chargeait des poids
        # On va adapter pour récupérer le modèle directement si possible, sinon on le recharge
        self.video_model = torch.load(Config.VIDEO_MODEL_PATH, map_location=self.device)
        if hasattr(self.video_model, 'classifier'): # Si c'est un modèle complet
            self.video_model.classifier = nn.Identity() # On veut les features (1280), pas la classif
        self.video_model.eval()

    def _start_text_server(self):
        print("🔧 Démarrage Serveur Texte (Llama)...")
        cfg = Config.TEXT_ENCODER_CONFIG
        # On lance le serveur s'il n'est pas déjà lancé manuellement
        self.text_process = start_server(cfg['executable'], cfg['model'], cfg['port'])
        time.sleep(5) # Attente chauffe serveur

    def _load_model(self):
        print("🧠 Chargement du Cerveau (TRM)...")
        # Doit matcher exactement train.py
        config = TinyRecursiveReasoningModel_ACTV1Config(
            batch_size=1, 
            input_dim=Config.INPUT_DIM,
            num_classes=Config.NUM_CLASSES,
            H_cycles=2, 
            L_cycles=1, # Config train.py
            L_layers=Config.L_LAYERS, 
            hidden_size=Config.HIDDEN_SIZE, 
            expansion=Config.EXPANSION, 
            num_heads=Config.NUM_HEADS, 
            pos_encodings="rope", 
            halt_max_steps=1,
            halt_exploration_prob=0.0, # Pas d'exploration en inférence
            no_ACT_continue=True, 
            forward_dtype="float32"
        )
        
        self.model = TinyRecursiveReasoningModel_ACTV1(config).to(self.device)
        
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"❌ ERREUR CRITIQUE: Modèle introuvable à {Config.MODEL_SAVE_PATH}")
            print("   Lancez 'train.py' d'abord.")
            sys.exit(1)
            
        # Chargement des poids
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def _init_hardware(self):
        print("📹 Ouverture Caméra & Micro...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Erreur: Caméra non détectée.")
            sys.exit(1)
            
        # Configuration Audio
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            blocksize=2048, # Petits paquets pour réactivité
            callback=self._audio_callback
        )

    def _audio_callback(self, indata, frames, time, status):
        """Callback exécuté par le thread audio system"""
        if status: print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def process_inputs(self):
        """Récupère et prépare les vecteurs Audio/Vidéo/Texte"""
        
        # 1. AUDIO (Hack: Sauvegarde temporaire car l'encodeur attend un fichier)
        # On récupère tout ce qui est dans la queue jusqu'à avoir environ 1 seconde
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.extend(self.audio_queue.get().flatten())
        
        # On garde seulement les dernières N secondes pour l'embedding
        max_samples = int(SAMPLE_RATE * BUFFER_DURATION)
        if len(audio_data) > max_samples:
            audio_chunk = np.array(audio_data[-max_samples:])
        elif len(audio_data) > 1000: # Au moins un peu de son
            audio_chunk = np.array(audio_data)
        else:
            return None # Pas assez de données

        # Sauvegarde temporaire pour l'encodeur existant
        temp_wav = "temp_inference.wav"
        # Normalisation float -> int16
        write_wav(temp_wav, SAMPLE_RATE, (audio_chunk * 32767).astype(np.int16))
        
        with torch.no_grad():
            # (1, T, 768) -> (768,) Moyenne temporelle pour ce step
            aud_emb = get_audio_embedding(self.audio_model, temp_wav, self.device)
            aud_emb = aud_emb.mean(dim=1).squeeze(0) # (768)
        
        # Nettoyage
        if os.path.exists(temp_wav): os.remove(temp_wav)

        # 2. VIDEO
        if not self.video_buffer: return None
        # On prend la dernière frame valide
        frame_rgb = self.video_buffer[-1]
        
        # Preprocessing Torchvision
        pil_img = Image.fromarray(frame_rgb)
        img_tensor = self.video_transform(pil_img).unsqueeze(0).to(self.device) # (1, 3, 224, 224)
        
        with torch.no_grad():
            vid_emb = self.video_model(img_tensor).squeeze(0) # (1280)

        # 3. TEXTE (Contexte situationnel simulé ou réel)
        # Pour l'instant, on met une phrase neutre ou contextuelle
        # Idéalement : utiliser Whisper ici pour transcrire l'audio_chunk
        text = "user is facing the camera looking at the screen" 
        txt_list = get_batch_embeddings([text], Config.TEXT_ENCODER_CONFIG['port'])
        
        if txt_list and len(txt_list) > 0:
            txt_emb = torch.tensor(txt_list[0], dtype=torch.float32).to(self.device) # (768)
        else:
            txt_emb = torch.zeros(768).to(self.device)

        # 4. FUSION (768 + 1280 + 768 = 2816)
        combined = torch.cat([aud_emb, vid_emb, txt_emb], dim=0).unsqueeze(0).unsqueeze(0) # (1, 1, 2816)
        
        # Normalisation (Critique !)
        combined = self.input_normalizer(combined)
        
        return combined

    def run(self):
        self.is_running = True
        self.stream.start()
        
        print("\n🟢 SYSTÈME PRÊT. Appuyez sur 'q' pour quitter.\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                
                # Conversion BGR -> RGB pour le modèle
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_buffer.append(frame_rgb)
                if len(self.video_buffer) > 10: self.video_buffer.pop(0)

                # --- PRÉDICTION PÉRIODIQUE ---
                now = time.time()
                if now - self.last_pred_time > PREDICTION_INTERVAL_S:
                    inputs = self.process_inputs()
                    
                    if inputs is not None:
                        with torch.no_grad():
                            # Init mémoire si première fois
                            if self.trm_carry is None:
                                # Le modèle attend inputs: (Batch, Dim) pour init
                                self.trm_carry = self.model.initial_carry({"inputs": inputs[:, 0, :]})
                            
                            # Forward step
                            self.trm_carry, out = self.model(self.trm_carry, {"inputs": inputs[:, 0, :]})
                            
                            # Récupération logits
                            logits = out['logits'].cpu().numpy()[0] # [Valence, Arousal]
                            
                            # Lissage pour éviter que ça saute partout
                            self.val_pred = 0.7 * self.val_pred + 0.3 * logits[0]
                            self.aro_pred = 0.7 * self.aro_pred + 0.3 * logits[1]
                            
                        self.last_pred_time = now

                # --- VISUALISATION ---
                # 1. Dessiner le cadre Valence/Arousal
                h, w, _ = frame.shape
                
                # Coordonnées (x, y) normalisées [-1, 1] -> écran
                cx, cy = int(w/2), int(h/2)
                # On map Valence (-1..1) -> X et Arousal (-1..1) -> Y (inverse car Y descend en image)
                pt_x = int(cx + (self.val_pred * 200)) # Scale 200px
                pt_y = int(cy - (self.aro_pred * 200))
                
                # Grille
                cv2.line(frame, (cx, cy-200), (cx, cy+200), (200, 200, 200), 1)
                cv2.line(frame, (cx-200, cy), (cx+200, cy), (200, 200, 200), 1)
                cv2.rectangle(frame, (cx-200, cy-200), (cx+200, cy+200), (255, 255, 255), 2)
                
                # Point de prédiction
                color = (0, 0, 255) # Rouge
                if self.val_pred > 0 and self.aro_pred > 0: color = (0, 255, 255) # Jaune (Heureux)
                elif self.val_pred < 0 and self.aro_pred < 0: color = (255, 0, 0) # Bleu (Triste/Calme)
                
                cv2.circle(frame, (pt_x, pt_y), 10, color, -1)
                
                # Texte
                info = f"Val: {self.val_pred:.2f} | Aro: {self.aro_pred:.2f}"
                cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('TRM Emotion AI', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n🛑 Arrêt du système...")
        self.stream.stop()
        self.stream.close()
        self.cap.release()
        cv2.destroyAllWindows()
        if self.text_process:
            self.text_process.terminate()

if __name__ == "__main__":
    app = RealTimeEmotionTester()
    app.run()