# Affective-TRM : Reconnaissance d'Émotion Multimodale Récursive

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red) ![Status](https://img.shields.io/badge/Status-Research_Prototype-purple) ![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)

**Affective-TRM** est une architecture de Deep Learning conçue pour la reconnaissance d'émotions en continu (**Valence / Arousal**) à partir de flux vidéo temps réel.

Contrairement aux approches classiques qui classifient une émotion en catégories discrètes (ex: "Colère", "Joie"), ce modèle projette l'état émotionnel dans un **espace latent continu** en fusionnant trois modalités : **Audio, Vidéo et Texte**.

Le modèle est capable de tourner en temps réel avec une empreinte mémoire réduite grâce à son architecture récursive.

---

## 🧠 Architecture du Modèle

Le cœur du projet repose sur le **Tiny Recursive Reasoning Model (TRM)**. C'est une architecture hybride inspirée des travaux de Samsung SAIL, combinant la puissance des Transformers avec l'efficacité séquentielle des RNNs.

<p align="center">
  <img src="docs/arch.png" alt="Architecture TRM" width="300"/>
</p>


*Architecture du Tiny Recursive Model (Source: Samsung SAIL)*

### 1. Entrée Multimodale Massive (2816 dims)
Le modèle ingère un vecteur concaténé représentant l'état complet de l'utilisateur à un instant $t$ :
*   **Audio (768 dims)** : Features extraites via **Dasheng** (Wav2Vec/Hubert optimisé).
*   **Vidéo (1280 dims)** : Features spatiales issues d'**EfficientNet-B0** pré-entraîné sur AFEW (via HSEmotion).
*   **Texte (768 dims)** : Embeddings sémantiques issus de **EmbeddingGemma** (Google).
*   *Fusion* : Les modalités sont concaténées horizontalement et normalisées (LayerNorm).

### 2. Le Flux de Données (Pipeline)
![Pipeline](docs/pipeline.png)

*   **Synchronisation** : Le système est cadencé sur la vidéo (30 FPS).
*   **Audio** : Alignement temporel par fenêtres glissantes synchronisées avec les frames.
*   **Texte** : Injection continue du contexte sémantique (dupliqué temporellement).
*   **Récurrence** : À chaque pas de temps, le TRM met à jour sa mémoire latente $Z$ en fonction de l'entrée $X_t$ et de son état précédent $Z_{t-1}$.

---

## 🔬 Méthodologie et Entraînement

Ce projet transforme un dataset de classification (CREMA-D) en problème de régression continue.

### 1. Mapping Discret $\rightarrow$ Continu
Les émotions discrètes sont projetées sur le modèle Circumplex de Russell via des centroïdes, avec ajout de **bruit gaussien** et modulation d'intensité pour simuler une distribution réelle :
*   **Colère** $\rightarrow$ Valence Négative / Arousal Haut
*   **Joie** $\rightarrow$ Valence Positive / Arousal Haut
*   **Tristesse** $\rightarrow$ Valence Négative / Arousal Bas
*   *Etc.*

### 2. Fonction de Perte (Loss) Hybride
Pour garantir à la fois la compréhension de la dynamique émotionnelle et la stabilité des prédictions, nous utilisons une Loss combinée :

$$Loss_{total} = Loss_{CCC} + 0.25 \times MSE_{real}$$

*   **$Loss_{CCC}$ ($1 - CCC$)** : Maximise la corrélation de forme. Force le modèle à comprendre les variations (montée/descente) de l'émotion.
*   **$MSE_{real}$ (Mean Squared Error)** : Agit comme une ancre pour empêcher les prédictions de dériver hors de l'espace $[-1, 1]$ et maintient l'échelle correcte.

---

## 📊 Résultats

Le modèle a été entraîné sur 25 époques avec une séparation stricte par locuteur (Speaker Independent).

| Métrique | Score Final (Ep 25) | Interprétation |
| :--- | :--- | :--- |
| **Train Loss** | 0.804 | Bonne convergence de l'apprentissage. |
| **Val Loss** | 0.839 | Pas d'overfitting majeur (courbe stable). |
| **Val CCC** | **0.255** | Corrélation positive significative sur des données synthétiques bruitées. |

### Visualisations
| Historique d'Apprentissage | Espace Valence/Arousal (Test) |
| :---: | :---: |
| ![History](docs/training_history.png) | ![AV Space](docs/av_space_plot.png) |
| *La dynamique du CCC (courbe verte) montre une amélioration constante jusqu'à la fin.* | *Le modèle (rouge) couvre correctement les quadrants haut définis par la vérité terrain (bleu), mais se retrouve en difficulté pour la partie basse.* |

---

## 🚀 Installation et Utilisation

### Pré-requis
*   Python 3.11+
*   PyTorch (CUDA recommandé)
*   `uv` ou `pip`

### 1. Téléchargement des Dépendances
Ce script télécharge automatiquement le dataset, les poids des encodeurs (Llama, Gemma, ENet) et les outils nécessaires.
```bash
python downloads.py
```

### 2. Préparation des Données
Extrait les features Audio/Vidéo/Texte et crée le dataset `.pt`.
```bash
python prepare_data.py
```

### 3. Entraînement (Optionnel)
Si vous souhaitez ré-entraîner le modèle depuis zéro :
```bash
python train.py
```

### 4. Inférence Temps Réel
Lance la webcam, enregistre l'audio et affiche l'émotion en direct.
```bash
python run_inference.py
```

---

## 📥 Modèle Pré-entraîné

Les poids du modèle entraîné (TRM) sont disponibles sur HuggingFace :
🤗 **[HuggingFace: JusteLeo/AffectiveTRM](https://huggingface.co/JusteLeo/AffectiveTRM)**

Téléchargez `emotion_model_sequential_av.pth` et placez-le à la racine du projet si vous ne voulez pas lancer l'entraînement.

---

## 📜 Crédits et Références

Ce projet n'aurait pas été possible sans les travaux de recherche et les modèles open-source suivants :

*   **Architecture TRM** : Basé sur *Tiny Recursive Models* (Samsung SAIL Montreal).
    *   [GitHub Repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
*   **Encodeur Audio** : *Dasheng* (DCASE 2023 Winner).
    *   [GitHub Repository](https://github.com/RicherMans/Dasheng)
*   **Encodeur Vidéo** : *HSEmotion* (EfficientNet-B0 sur AFEW).
    *   [GitHub Repository](https://github.com/av-savchenko/hsemotion)
*   **Encodeur Texte** : *EmbeddingGemma* (Google).
    *   [HuggingFace Model](https://huggingface.co/google/embeddinggemma-300m)
*   **Dataset** : *CREMA-D* (Crowd-sourced Emotional Multimodal Actors Dataset).
    *   [GitHub Repository](https://github.com/CheyneyComputerScience/CREMA-D)
---

*Projet réalisé dans le cadre d'un Master 1 Électronique / IA.*