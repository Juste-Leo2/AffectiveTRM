# train.py
import os
import torch
import random
import numpy as np
from src.config import Config
from src.data.dataset import create_dataloaders
from src.models.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from src.training.engine import train_model, save_model
from src.training.visualizer import plot_history_regression, plot_av_space

def set_seed(seed=42):
    """Fixe la seed pour tous les modules aléatoires."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Pour garantir la reproductibilité totale sur GPU (peut ralentir un peu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fixée à {seed} (Random, Numpy, Torch, Cuda).")

def main():
    # --- REPRODUCTIBILITÉ ---
    set_seed(42)

    if not os.path.exists(Config.DATA_PROCESSED_PATH):
        print("Erreur: Données introuvables. Lancez 'prepare_data.py'.")
        return

    print(f"\n--- Démarrage Entraînement sur {Config.DEVICE} ---")
    
    # Configuration du modèle
    model_config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=Config.BATCH_SIZE, 
        input_dim=Config.INPUT_DIM, # 2816 maintenant
        num_classes=Config.NUM_CLASSES,
        
        H_cycles=2, 
        L_cycles=1, 
        
        L_layers=Config.L_LAYERS, 
        hidden_size=Config.HIDDEN_SIZE, 
        expansion=Config.EXPANSION, 
        num_heads=Config.NUM_HEADS, 
        pos_encodings="rope", 
        halt_max_steps=1, 
        
        # --- AJOUT DU CHAMP MANQUANT ---
        halt_exploration_prob=0.05, 
        
        no_ACT_continue=True, 
        forward_dtype="float32"
    )

    # Données
    train_loader, val_loader, test_loader = create_dataloaders(Config.DATA_PROCESSED_PATH, Config.BATCH_SIZE)

    # Modèle
    model = TinyRecursiveReasoningModel_ACTV1(model_config).to(Config.DEVICE)
    print(f"Paramètres modèle : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train
    history = train_model(model, train_loader, val_loader, Config.NUM_EPOCHS, Config.DEVICE)

    # Visualisation (Sauvegarde au lieu d'affichage)
    plot_history_regression(history)
    try:
        plot_av_space(model, test_loader, Config.DEVICE)
    except Exception as e:
        print(f"Erreur Plot: {e}")
        
    save_model(model, Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()