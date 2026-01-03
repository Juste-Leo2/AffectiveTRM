# src/training/visualizer.py
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm

def plot_history_regression(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique 1 : La Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Combined Loss (Lower is Better)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    
    # Graphique 2 : Le Score CCC
    if 'val_ccc' in history:
        ax2.plot(history['val_ccc'], label='Validation CCC', color='green')
        ax2.set_title('Concordance Correlation Coeff (Higher is Better)')
        ax2.set_ylabel('CCC Score')
    elif 'val_mae' in history:
        ax2.plot(history['val_mae'], label='Validation MAE', color='orange')
        ax2.set_title('Mean Absolute Error (Lower is Better)')
        ax2.set_ylabel('MAE')
        
    ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)
    
    # SAUVEGARDE
    save_path = "training_history.png"
    plt.savefig(save_path)
    plt.close() # Ferme la figure pour libérer la mémoire
    print(f"Graphique d'historique sauvegardé sous : {os.path.abspath(save_path)}")

def plot_av_space(model, test_loader, device):
    model.eval()
    all_preds, all_true = [], []
    
    print("Génération du graphique Valence/Arousal...")
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing for AV plot"):
            # Gestion robuste du déballage
            if len(batch_data) == 4:
                sequences, labels_av, _, _ = batch_data
            else:
                sequences, labels_av, _ = batch_data
                
            sequences = sequences.to(device)
            
            # Init mémoire
            batch_for_init = {"inputs": sequences[:, 0, :]}
            carry = model.initial_carry(batch_for_init)
            
            outputs = None
            # Boucle temporelle
            for t in range(sequences.size(1)):
                carry, outputs = model(carry, {"inputs": sequences[:, t, :]})
            
            # FILTRAGE
            valid_mask = labels_av[:, 0] > -1.5
            
            if valid_mask.any():
                all_preds.append(outputs['logits'][valid_mask].cpu())
                all_true.append(labels_av[valid_mask].cpu())
    
    if not all_preds:
        print("Attention: Aucune donnée avec intensité spécifiée trouvée. Graphique annulé.")
        return

    preds_np = torch.cat(all_preds).numpy()
    true_np = torch.cat(all_true).numpy()
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # Limites fixes [-1, 1]
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    
    # Centrer les axes
    ax.spines['left'].set_position('center'); ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom'); ax.yaxis.set_ticks_position('left')
    
    plt.xlabel("Valence (X)", loc='right'); plt.ylabel("Arousal (Y)", loc='top')
    plt.title("Prédictions (rouge) vs Cibles (bleu)")
    
    # Limite d'affichage
    limit = 300
    if len(preds_np) > limit:
        # On utilise une seed locale ici aussi pour que le sous-échantillonnage soit le même
        rng = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(preds_np), generator=rng)[:limit]
        preds_np = preds_np[indices]
        true_np = true_np[indices]
        
    # Dessiner les flèches (erreurs)
    for i in range(len(preds_np)):
        plt.arrow(true_np[i, 0], true_np[i, 1], 
                  preds_np[i, 0] - true_np[i, 0], preds_np[i, 1] - true_np[i, 1], 
                  color='gray', alpha=0.3, head_width=0.02)
                  
    plt.scatter(true_np[:, 0], true_np[:, 1], c='blue', label='Ground Truth', alpha=0.6)
    plt.scatter(preds_np[:, 0], preds_np[:, 1], c='red', label='Prediction', alpha=0.6)
    plt.legend(loc='upper right'); plt.grid()
    
    # SAUVEGARDE
    save_path = "av_space_plot.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Graphique Valence/Arousal sauvegardé sous : {os.path.abspath(save_path)}")