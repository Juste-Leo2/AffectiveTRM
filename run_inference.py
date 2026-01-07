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

# --- PARAMÈTRES ---
PREDICTION_INTERVAL_S = 0.5
SAMPLE_RATE = 16000
VIDEO_FPS = 30
BUFFER_DURATION = 1.0 

# Phrase neutre d'ancrage (issue du training set)
NEUTRAL_SENTENCE = "It's eleven o'clock"

class RealTimeEmotionTester:
    def __init__(self):
        # Force CPU si demandé (plus stable pour la démo si pas de GPU dédié)
        self.device = torch.device("cpu") # Ou Config.DEVICE si tu veux retenter le GPU
        print(f"\n📱 Démarrage Inference sur : {self.device}")
        
        # Initialisation Détection Visage (Natif OpenCV, très rapide)
        haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
        
        self._init_transforms()
        self._load_encoders()
        
        # --- NOUVEAU : Pré-calcul du texte ---
        self._init_text_cache()
        
        self._load_model()
        self._init_hardware()
        
        self.audio_queue = queue.Queue()
        self.video_buffer = []
        self.is_running = False
        self.last_pred_time = 0
        
        self.val_pred = 0.0
        self.aro_pred = 0.0
        self.trm_carry = None

    def _init_transforms(self):
        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.input_normalizer = nn.LayerNorm(Config.INPUT_DIM).to(self.device)

    def _load_encoders(self):
        print("🔧 Chargement des encodeurs...")
        # Audio
        self.audio_model = audio_encoder_init()
        self.audio_model.to(self.device).eval()
        
        print(f"   Chargement modèle vidéo (Reconstruction timm)...")
        import timm
        # Reconstruction propre de l'architecture pour éviter les bugs de version
        self.video_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=8)
        
        try:
            checkpoint = torch.load(Config.VIDEO_MODEL_PATH, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Erreur chargement fichier vidéo : {e}")
            sys.exit(1)

        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            try: state_dict = checkpoint.state_dict()
            except: state_dict = checkpoint

        self.video_model.load_state_dict(state_dict, strict=False)
        self.video_model.classifier = nn.Identity() # On garde les features (1280)
        self.video_model.to(self.device).eval()

    def _init_text_cache(self):
        """
        Lance le serveur, calcule l'embedding de la phrase neutre, et éteint le serveur.
        Économise énormément de CPU/RAM.
        """
        print(f"🔧 Pré-calcul du texte neutre : '{NEUTRAL_SENTENCE}'...")
        cfg = Config.TEXT_ENCODER_CONFIG
        
        # 1. Démarrage temporaire
        proc = start_server(cfg['executable'], cfg['model'], cfg['port'])
        time.sleep(5) # Attente chauffe
        
        # 2. Calcul
        try:
            res = get_batch_embeddings([NEUTRAL_SENTENCE], cfg['port'])
            if res and len(res) > 0:
                self.cached_text_emb = torch.tensor(res[0], dtype=torch.float32).to(self.device)
                print("   ✅ Embedding texte mis en cache.")
            else:
                print("   ⚠️ Erreur calcul texte, utilisation de bruit aléatoire (moins bon).")
                self.cached_text_emb = torch.randn(768).to(self.device)
        except Exception as e:
            print(f"   ⚠️ Erreur serveur: {e}")
            self.cached_text_emb = torch.zeros(768).to(self.device)
        
        # 3. Arrêt du serveur (Optimisation CPU)
        if proc:
            print("   🛑 Arrêt du serveur de texte pour libérer les ressources.")
            proc.terminate()
            proc.wait()
        
        self.text_process = None # On marque comme fermé

    def _load_model(self):
        print("🧠 Chargement TRM...")
        config = TinyRecursiveReasoningModel_ACTV1Config(
            batch_size=1, input_dim=Config.INPUT_DIM, num_classes=Config.NUM_CLASSES,
            H_cycles=2, L_cycles=1, L_layers=Config.L_LAYERS, 
            hidden_size=Config.HIDDEN_SIZE, expansion=Config.EXPANSION, 
            num_heads=Config.NUM_HEADS, pos_encodings="rope", 
            halt_max_steps=1, halt_exploration_prob=0.0, 
            no_ACT_continue=True, forward_dtype="float32"
        )
        self.model = TinyRecursiveReasoningModel_ACTV1(config).to(self.device)
        
        # Chargement poids
        try:
            # weights_only=True est le défaut maintenant, mais on force False si besoin selon pytorch version
            # Ici pour un state_dict simple, True marche souvent, mais restons safe
            ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            self.model.load_state_dict(ckpt)
        except Exception:
            ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt)
            
        self.model.eval()

    def _init_hardware(self):
        self.cap = cv2.VideoCapture(0)
        # Augmentation du blocksize audio pour éviter les 'input overflow' sur CPU chargé
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=8192, callback=self._audio_callback)

    def _audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(indata.copy())

    def get_face_from_frame(self, frame_rgb):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0: return None
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        margin = int(h * 0.2)
        y1 = max(0, y - margin)
        y2 = min(frame_rgb.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(frame_rgb.shape[1], x + w + margin)
        return frame_rgb[y1:y2, x1:x2]

    def process_inputs(self):
        # 1. AUDIO
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.extend(self.audio_queue.get().flatten())
        
        target_len = int(SAMPLE_RATE * BUFFER_DURATION)
        if len(audio_data) < 2000: return None # Un peu plus strict sur la quantité d'audio
        
        chunk = np.array(audio_data[-target_len:] if len(audio_data) > target_len else audio_data)
        temp_wav = "temp_live.wav"
        write_wav(temp_wav, SAMPLE_RATE, (chunk * 32767).astype(np.int16))
        
        with torch.no_grad():
            aud_emb = get_audio_embedding(self.audio_model, temp_wav, self.device).mean(dim=1).squeeze(0)
        
        # 2. VIDEO (Avec Crop)
        if not self.video_buffer: return None
        face_img = self.get_face_from_frame(self.video_buffer[-1])
        if face_img is None: return None
            
        pil_img = Image.fromarray(face_img)
        img_tensor = self.video_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vid_emb = self.video_model(img_tensor).squeeze(0)

        # 3. TEXTE (Depuis le CACHE)
        txt_emb = self.cached_text_emb

        # 4. FUSION
        combined = torch.cat([aud_emb, vid_emb, txt_emb], dim=0).unsqueeze(0).unsqueeze(0)
        return self.input_normalizer(combined)

    def run(self):
        self.is_running = True
        self.stream.start()
        print("\n🟢 LIVE (Mode CPU Optimisé). 'q' pour quitter.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_buffer.append(frame_rgb)
                if len(self.video_buffer) > 3: self.video_buffer.pop(0) # Buffer court pour latence min

                now = time.time()
                if now - self.last_pred_time > PREDICTION_INTERVAL_S:
                    try:
                        inputs = self.process_inputs()
                        if inputs is not None:
                            with torch.no_grad():
                                if self.trm_carry is None:
                                    self.trm_carry = self.model.initial_carry({"inputs": inputs[:, 0, :]})
                                self.trm_carry, out = self.model(self.trm_carry, {"inputs": inputs[:, 0, :]})
                                logits = out['logits'].cpu().numpy()[0]
                                
                                val_raw = np.clip(logits[0], -1.0, 1.0)
                                aro_raw = np.clip(logits[1], -1.0, 1.0)
                                
                                # Lissage un peu plus lent pour stabilité visuelle
                                self.val_pred = 0.7 * self.val_pred + 0.3 * val_raw
                                self.aro_pred = 0.7 * self.aro_pred + 0.3 * aro_raw
                                
                            self.last_pred_time = now
                    except Exception:
                        pass # On ignore les erreurs de frames drop pour fluidité

                # --- UI ---
                # On dessine sur l'image originale BGR
                
                # Petit indicateur de visage
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 100), 1)

                # Graphe
                h, w, _ = frame.shape
                cx, cy = 90, h - 90
                size = 70
                
                # Fond semi-transparent
                overlay = frame.copy()
                cv2.rectangle(overlay, (cx-size-10, cy-size-10), (cx+size+10, cy+size+10), (240, 240, 240), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Axes
                cv2.line(frame, (cx, cy-size), (cx, cy+size), (50,50,50), 1)
                cv2.line(frame, (cx-size, cy), (cx+size, cy), (50,50,50), 1)
                
                # Quadrants labels (discrets)
                cv2.putText(frame, "Pos", (cx+size-25, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)
                cv2.putText(frame, "Neg", (cx-size+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)
                cv2.putText(frame, "High", (cx+2, cy-size+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)
                cv2.putText(frame, "Low", (cx+2, cy+size-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)

                # Point courant
                px = int(cx + (self.val_pred * size))
                py = int(cy - (self.aro_pred * size))
                
                # Couleur dynamique du point
                # Rouge (Colère/High Neg) -> Jaune (Joie/High Pos) -> Bleu (Calme/Low)
                color = (0, 0, 255) # Defaut rouge
                if self.val_pred > 0.2: color = (0, 200, 255) # Jaune/Orange
                if self.aro_pred < -0.2: color = (255, 100, 0) # Bleu foncé
                
                cv2.circle(frame, (px, py), 8, color, -1)
                cv2.circle(frame, (px, py), 9, (50,50,50), 1) # Contour noir

                cv2.imshow('Affective TRM - CPU Mode', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        finally:
            self.cleanup()

    def cleanup(self):
        print("Nettoyage...")
        try: self.stream.stop(); self.stream.close()
        except: pass
        self.cap.release()
        cv2.destroyAllWindows()
        # Le text process est déjà fermé, mais on check au cas où
        if self.text_process: 
            try: self.text_process.terminate()
            except: pass

if __name__ == "__main__":
    RealTimeEmotionTester().run()