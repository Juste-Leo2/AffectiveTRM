# src/training/engine.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from src.config import Config

# Centres de gravité (Guidage)
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
        ccc = (2 * cov * x_std * y_std) / (x.var() + y.var() + (x_m - y_m)**2 + 1e-8)
        return 1.0 - ccc

def criterion_combined(preds, targets_av, targets_class, device):
    # 1. CCC Loss
    valid = (targets_av[:, 0] > -1.5)
    ccc = torch.tensor(0.0, device=device)
    if valid.any():
        p, t = preds[valid], targets_av[valid]
        if len(p) >= 2:
            loss_fn = CCCLoss()
            ccc = (loss_fn(p[:,0], t[:,0]) + loss_fn(p[:,1], t[:,1])) / 2.0
        else: ccc = F.mse_loss(p, t)
    
    # 2. Zone Loss
    centers = CENTROIDS.to(device)[targets_class]
    zone = F.mse_loss(preds, centers)
    
    return ccc + 0.0 * zone, ccc, zone

def train_model(model, train_loader, val_loader, num_epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # --- IMPORTANT : Mise à jour de la dimension ---
    input_normalizer = nn.LayerNorm(Config.INPUT_DIM).to(device)

    history = {'train_loss': [], 'val_loss': [], 'val_ccc': []}
    
    print(f"\n--- Train Start (Last Step + Zone) - Input Dim: {Config.INPUT_DIM} ---")
    
    for epoch in range(num_epochs):
        model.train()
        t_loss, t_ccc, t_zone, n_batches = 0, 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for seq, lbl_av, lbl_cls, _ in pbar:
            seq, lbl_av, lbl_cls = seq.to(device), lbl_av.to(device), lbl_cls.to(device)
            seq = input_normalizer(seq)
            
            optimizer.zero_grad()
            carry = model.initial_carry({"inputs": seq[:, 0, :]})
            
            # Forward complet
            for t in range(seq.size(1)):
                carry, out = model(carry, {"inputs": seq[:, t, :]})
            
            # Loss finale
            loss, ccc, zone = criterion_combined(out['logits'], lbl_av, lbl_cls, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            t_loss += loss.item(); t_ccc += (ccc.item() if hasattr(ccc, 'item') else ccc)
            t_zone += zone.item(); n_batches += 1
            pbar.set_postfix(L=f"{loss.item():.2f}", C=f"{t_ccc/n_batches:.2f}")

        avg_loss = t_loss / (n_batches + 1e-8)
        history['train_loss'].append(avg_loss)
        
        # Validation
        val_loss, val_ccc = evaluate(model, val_loader, input_normalizer, device)
        history['val_loss'].append(val_loss)
        history['val_ccc'].append(val_ccc)
        
        scheduler.step(val_loss)
        print(f"Ep {epoch+1} | Train Loss: {avg_loss:.3f} | Val CCC: {val_ccc:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

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
            
            l, c, _ = criterion_combined(out['logits'], lbl_av, lbl_cls, device)
            t_loss += l.item(); t_ccc += (c.item() if hasattr(c, 'item') else c); n += 1
    return t_loss/(n+1e-8), t_ccc/(n+1e-8)

def save_model(model, path): torch.save(model.state_dict(), path)