# src/training/engine.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from src.config import Config

# Centres de gravité (Utilisés uniquement pour l'affichage ou debug désormais)
CENTROIDS = torch.tensor([
    [0.0, 0.0],   [0.7, 0.6],   [-0.7, -0.5],
    [-0.6, 0.7],  [-0.3, 0.8],  [-0.7, 0.3]
], dtype=torch.float32)

class CCCLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, y):
        x_m, y_m = x.mean(), y.mean()
        cov = ((x - x_m) * (y - y_m)).mean()
        x_std, y_std = x.std(), y.std()
        var_x = x.var()
        var_y = y.var()
        
        # Petit epsilon pour éviter la division par zéro
        ccc = (2 * cov * x_std * y_std) / (var_x + var_y + (x_m - y_m)**2 + 1e-8)
        return 1.0 - ccc

def criterion_combined(preds, targets_av, targets_class, device):
    """
    Kombination : CCC (Corrélation) + MSE Réel (Précision)
    """
    # On filtre les données valides (au cas où il y a du padding -2)
    valid = (targets_av[:, 0] > -1.5)
    
    ccc_loss = torch.tensor(0.0, device=device)
    mse_real_loss = torch.tensor(0.0, device=device)
    
    if valid.any():
        p, t = preds[valid], targets_av[valid]
        
        # 1. Calcul du CCC (Gère la forme du nuage / la direction)
        if len(p) >= 2:
            loss_fn = CCCLoss()
            # Moyenne du CCC Valence et du CCC Arousal
            ccc_loss = (loss_fn(p[:,0], t[:,0]) + loss_fn(p[:,1], t[:,1])) / 2.0
        else:
            # Fallback si batch trop petit
            ccc_loss = F.mse_loss(p, t)
            
        # 2. Calcul du MSE Réel (Gère l'échelle et empêche les points de partir trop loin)
        mse_real_loss = F.mse_loss(p, t)
    
    # 3. Calcul Zone (Optionnel, juste pour info, ne participe pas au gradient)
    # centers = CENTROIDS.to(device)[targets_class]
    # zone_loss = F.mse_loss(preds, centers)
    
    # --- FORMULE FINALE ---
    # On met un poids de 0.25 sur le MSE.
    # C'est suffisant pour "calmer" les points, sans tuer la dynamique du CCC.
    total_loss = ccc_loss + 0.25 * mse_real_loss
    
    # On retourne : Loss Totale, La part CCC, La part MSE
    return total_loss, ccc_loss, mse_real_loss

def train_model(model, train_loader, val_loader, num_epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    input_normalizer = nn.LayerNorm(Config.INPUT_DIM).to(device)

    history = {'train_loss': [], 'val_loss': [], 'val_ccc': []}
    
    print(f"\n--- Train Start (CCC + 0.25 * RealMSE) - Input Dim: {Config.INPUT_DIM} ---")
    
    for epoch in range(num_epochs):
        model.train()
        t_loss, t_ccc, t_mse, n_batches = 0, 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for seq, lbl_av, lbl_cls, _ in pbar:
            seq, lbl_av, lbl_cls = seq.to(device), lbl_av.to(device), lbl_cls.to(device)
            seq = input_normalizer(seq)
            
            optimizer.zero_grad()
            carry = model.initial_carry({"inputs": seq[:, 0, :]})
            
            for t in range(seq.size(1)):
                carry, out = model(carry, {"inputs": seq[:, t, :]})
            
            # Appel de la nouvelle loss
            loss, ccc_part, mse_part = criterion_combined(out['logits'], lbl_av, lbl_cls, device)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Suivi des métriques
            t_loss += loss.item()
            t_ccc += ccc_part.item() if hasattr(ccc_part, 'item') else ccc_part
            t_mse += mse_part.item() if hasattr(mse_part, 'item') else mse_part
            n_batches += 1
            
            # Affichage barre : L = Loss Totale, C = CCC Part (plus c'est bas, mieux c'est)
            pbar.set_postfix(L=f"{loss.item():.3f}", C=f"{ccc_part.item():.3f}")

        avg_loss = t_loss / (n_batches + 1e-8)
        avg_ccc_loss = t_ccc / (n_batches + 1e-8)
        history['train_loss'].append(avg_loss)
        
        # Validation
        val_loss, val_ccc_loss = evaluate(model, val_loader, input_normalizer, device)
        history['val_loss'].append(val_loss)
        
        # Pour l'historique, on enregistre le vrai SCORE CCC (1 - loss) pour que le graphique monte
        history['val_ccc'].append(1.0 - val_ccc_loss)
        
        scheduler.step(val_loss)
        
        # Affichage Fin Epoch
        # Val CCC Score = 1.0 - val_ccc_loss (C'est plus lisible pour l'humain)
        print(f"Ep {epoch+1} | Train Loss: {avg_loss:.3f} | Val Loss: {val_loss:.3f} | Val CCC Score: {1.0 - val_ccc_loss:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    return history

def evaluate(model, loader, norm, device):
    model.eval()
    t_loss, t_ccc, n = 0, 0, 0
    with torch.no_grad():
        for seq, lbl_av, lbl_cls, _ in loader:
            seq, lbl_av, lbl_cls = seq.to(device), lbl_av.to(device), lbl_cls.to(device)
            seq = norm(seq)
            carry = model.initial_carry({"inputs": seq[:, 0, :]})
            for t in range(seq.size(1)): carry, out = model(carry, {"inputs": seq[:, t, :]})
            
            l, c, m = criterion_combined(out['logits'], lbl_av, lbl_cls, device)
            t_loss += l.item()
            t_ccc += c.item() if hasattr(c, 'item') else c
            n += 1
    return t_loss/(n+1e-8), t_ccc/(n+1e-8)

def save_model(model, path): torch.save(model.state_dict(), path)