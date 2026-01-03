# Affective-TRM : Reconnaissance d'Émotion Multimodale Récursive

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![Status](https://img.shields.io/badge/Status-Research_Prototype-purple)

**Affective-TRM** est une architecture expérimentale de Deep Learning conçue pour la reconnaissance d'émotions en continu (Valence / Arousal) à partir de flux vidéo. 

Contrairement aux approches classiques qui classifient une émotion en catégories discrètes (ex: "Colère", "Joie"), ce modèle projette l'état émotionnel dans un **espace latent continu** en fusionnant trois modalités : **Audio, Vidéo et Texte**.

## 🧠 Architecture du Modèle

Le cœur du projet repose sur le **Tiny Recursive Reasoning Model (TRM)**. C'est une architecture hybride qui combine la puissance des Transformers avec l'efficacité séquentielle des RNNs.

### Points Clés de l'Architecture :
1.  **Entrée Multimodale Massive (2816 dims)** :
    *   **Audio (768)** : Features extraites via Wav2Vec/Hubert.
    *   **Vidéo (1280)** : Features spatiales issues d'EfficientNet (enet_b0).
    *   **Texte (768)** : Embeddings sémantiques issus de LLM (Llama/Gemma).
    *   *Fusion* : Les modalités sont concaténées et normalisées (LayerNorm) avant d'entrer dans le réseau.

2.  **Transformer Récurrent (TRM)** :
    *   Au lieu de traiter toute la vidéo d'un coup (ce qui exploserait la VRAM), le modèle traite la séquence frame par frame.
    *   Il maintient une **mémoire persistante (Carry State)** composée de :
        *   $z_H$ (High-level) : Contexte global et émotionnel à long terme.
        *   $z_L$ (Low-level) : Mémoire de travail pour les calculs immédiats.
    *   À chaque pas de temps, l'input est injecté et fusionné avec la mémoire via des blocs d'attention (SwiGLU + RoPE).

3.  **Dual-Path Decision** :
    *   **Shortcut Head** : Une voie rapide qui permet au modèle de réagir aux signaux évidents (ex: un cri fort) immédiatement.
    *   **Deep Reasoning Head** : Une voie profonde qui analyse le contexte temporel stocké dans $z_H$ pour affiner la prédiction.

## 🔬 Méthodologie et Données

Ce projet utilise une méthodologie de transformation de données innovante pour convertir un dataset de classification (ex: CREMA-D) en problème de régression.

### 1. Mapping Discret vers Continu
Les émotions discrètes sont mappées sur l'espace Valence/Arousal (Modèle Circumplex de Russell) :
*   **Colère (ANG)** $\rightarrow$ Valence Négative / Arousal Haut
*   **Tristesse (SAD)** $\rightarrow$ Valence Négative / Arousal Bas
*   **Joie (HAP)** $\rightarrow$ Valence Positive / Arousal Haut
*   *Etc.*

### 2. Gestion de l'Intensité & Data Augmentation
Pour éviter que le modèle n'apprenne des points fixes par cœur, nous utilisons une stratégie de **Label Smoothing Spatial** :
*   Chaque émotion possède un centre de gravité théorique.
*   Ce centre est déplacé selon l'intensité annotée (`LO`, `MD`, `HI`).
*   Pour les intensités inconnues (`XX`), une intensité aléatoire est simulée.
*   Un **bruit gaussien** est ajouté à chaque échantillon.
*   **Résultat :** Le modèle doit apprendre à viser des "zones" émotionnelles plutôt que des coordonnées exactes, ce qui améliore considérablement la généralisation.

### 3. Fonction de Perte (Loss) Hybride
L'entraînement minimise une combinaison de deux pertes :
$$Loss = (1 - CCC) + \alpha \times MSE_{zone}$$
*   **CCC (Concordance Correlation Coefficient)** : Maximise la corrélation temporelle et l'accord d'amplitude.
*   **Zone Loss (MSE)** : Guide le modèle vers le bon quadrant émotionnel, crucial en début d'entraînement.

## 🚀 Installation et Utilisation

### Pré-requis
*   Python 3.10+
*   PyTorch avec support CUDA
*   `uv` (recommandé) ou `pip`

### 1. Préparation des Données
Le script de préparation scanne les fichiers bruts, extrait les embeddings (Audio/Vidéo/Texte) et génère un dataset `.pt` optimisé.

```bash
# Vérifiez les chemins dans src/config.py avant de lancer
uv run prepare_data.py
```
*Note : Cette étape peut être longue car elle effectue l'inférence des encodeurs (Audio/Vidéo/LLM).*

### 2. Entraînement
Lance la boucle d'entraînement avec validation croisée (Speaker Independent Split).

```bash
uv run train.py
```
Le script gère automatiquement :
*   La normalisation des entrées.
*   Le split Train/Val/Test (garantissant qu'un acteur n'est pas vu en train et en test).
*   La sauvegarde du meilleur modèle.
*   L'affichage des courbes de Loss et CCC.

### 3. Visualisation
À la fin de l'entraînement, deux graphiques sont générés :
1.  **Historique d'apprentissage** : Évolution de la Loss et du score CCC.
2.  **Espace Valence/Arousal** : Un scatter plot montrant les prédictions (rouge) vs la vérité terrain (bleu), permettant d'analyser la dynamique du modèle (ex: phénomène de régression vers la moyenne).

## 📊 Résultats Observés

Sur le dataset CREMA-D transformé :
*   **CCC Score** : ~0.77 (Performance état de l'art pour cette approche).
*   **Comportement** : Le modèle démontre une capacité robuste à distinguer les valences positives/négatives. Il adopte un comportement conservateur sur l'intensité (régression vers la moyenne), typique des approches par régression sur des données bruitées.

## 📂 Structure du Projet

```
.
├── src/
│   ├── config.py           # Configuration globale (Hyperparamètres, Chemins)
│   ├── data/
│   │   ├── dataset.py      # Gestion du Dataset PyTorch & Collate
│   │   └── preprocessor.py # Extraction des features & Mapping Émotionnel
│   ├── models/
│   │   ├── trm.py          # Architecture Tiny Recursive Model
│   │   └── layers.py       # Blocs de base (Attention, RMSNorm, SwiGLU)
│   └── training/
│       ├── engine.py       # Boucle d'entraînement & Fonctions de Loss
│       └── visualizer.py   # Outils de plotting (Matplotlib)
├── train.py                # Point d'entrée principal
├── prepare_data.py         # Script de pré-traitement
└── README.md
```

## 📜 Crédits
*   **Architecture TRM** : Inspirée des travaux sur les *Recurrent Transformers* et *Adaptive Computation Time*.
*   **Dataset** : Basé sur CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset).
*   **Encoders** : Utilise des poids pré-entraînés pour l'extraction de features (Wav2Vec2, EfficientNet, Gemma/Llama).