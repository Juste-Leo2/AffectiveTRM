# audio_encoder.py (Version modifiée pour support GPU)

# Librairies nécessaires: pip install dasheng torch torchaudio scipy numpy
import torch
import torchaudio
from dasheng import dasheng_base
from scipy.io.wavfile import read as read_wav 
import numpy as np

def audio_encoder_init():
    """
    Initialise et retourne le modèle d'encodage audio Dasheng.
    Le modèle est laissé sur le CPU par défaut, il sera déplacé plus tard.
    """
    model = dasheng_base()
    model.eval()
    return model

def get_audio_embedding(model, file_path, device):
    """
    Prend un modèle, le chemin d'un fichier .wav, et un device (ex: 'cuda')
    et retourne son embedding calculé sur ce device.
    """
    # 1. Charger le fichier audio avec Scipy
    original_sr, waveform_np = read_wav(file_path)

    # 2. Convertir en Tenseur PyTorch et normaliser (reste sur le CPU pour l'instant)
    waveform = torch.from_numpy(waveform_np.astype(np.float32) / 32767.0)

    # 3. Ré-échantillonner à 16kHz si nécessaire (sur CPU)
    if original_sr != 16000:
        resampler = torchaudio.transforms.Resample(original_sr, 16000)
        waveform = resampler(waveform)

    # 4. Assurer le bon format pour le modèle (batch, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
        
    # --- MODIFICATION CLÉ ---
    # 5. Déplacer le tenseur sur le bon device (CPU ou GPU)
    waveform = waveform.to(device)
    
    # 6. Obtenir l'embedding (le calcul se fera sur le device du modèle et des données)
    with torch.no_grad():
        embedding = model(waveform)
        
    # 7. Renvoyer l'embedding sur le CPU pour la suite du traitement
    return embedding.cpu()