# src/data/dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from src.config import Config 

class EmotionSequenceDataset(Dataset):
    def __init__(self, preprocessed_data_path):
        self.data = torch.load(preprocessed_data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_cls = item.get('label_class', torch.tensor(0, dtype=torch.long))
        
        # --- MODIFICATION ICI ---
        # On récupère la séquence
        seq = item['sequence_embeddings']
        
        # On coupe si c'est plus long que NUM_FRAMES (ex: 30)
        if seq.size(0) > Config.NUM_FRAMES:
            seq = seq[:Config.NUM_FRAMES, :]
            
        return seq, item['label_av'], label_cls

def collate_fn(batch):
    sequences, labels_av, labels_class = zip(*batch)
    
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels_av = torch.stack(labels_av)
    labels_class = torch.stack(labels_class)
    
    lengths = [len(s) for s in sequences]
    mask = torch.arange(padded_sequences.size(1))[None, :] < torch.tensor(lengths)[:, None]
    
    return padded_sequences, labels_av, labels_class, mask

def create_dataloaders(preprocessed_data_path, batch_size, split_ratios=(0.8, 0.1)):
    """
    Crée des DataLoaders avec une séparation stricte par locuteur (Speaker Independent).
    """
    print(f"Chargement du dataset depuis {preprocessed_data_path}...")
    full_dataset = EmotionSequenceDataset(preprocessed_data_path)
    
    # 1. Regrouper les indices par Acteur
    actor_indices = defaultdict(list)
    missing_actor_id = False
    
    for idx, item in enumerate(full_dataset.data):
        if 'actor_id' in item:
            actor_indices[item['actor_id']].append(idx)
        else:
            missing_actor_id = True
            # Fallback temporaire si l'utilisateur n'a pas mis à jour les données
            # On utilise un "fake" actor ID basé sur un hash pour essayer de grouper un peu
            # Mais idéalement, il faut relancer le prepare_data
            fake_id = idx % 91 
            actor_indices[fake_id].append(idx)
            
    if missing_actor_id:
        print("\n[ATTENTION] 'actor_id' manquant dans les données pré-traitées.")
        print("La séparation par locuteur ne sera pas optimale.")
        print("Veuillez relancer 'prepare_data.py' pour mettre à jour la structure des données.\n")

    # 2. Séparer les IDs d'acteurs
    unique_actors = list(actor_indices.keys())
    # Mélange des acteurs (seed fixe pour reproductibilité)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_actors)
    
    n_actors = len(unique_actors)
    n_train = int(split_ratios[0] * n_actors)
    n_val = int(split_ratios[1] * n_actors)
    
    train_actors = unique_actors[:n_train]
    val_actors = unique_actors[n_train:n_train+n_val]
    test_actors = unique_actors[n_train+n_val:]
    
    print(f"Total Acteurs: {n_actors}")
    print(f"Train Acteurs: {len(train_actors)} | Val Acteurs: {len(val_actors)} | Test Acteurs: {len(test_actors)}")
    
    # 3. Construire les listes d'indices finaux
    train_indices = []
    for act in train_actors: train_indices.extend(actor_indices[act])
        
    val_indices = []
    for act in val_actors: val_indices.extend(actor_indices[act])
        
    test_indices = []
    for act in test_actors: test_indices.extend(actor_indices[act])
    
    # Création des Subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Samples -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Création des Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader