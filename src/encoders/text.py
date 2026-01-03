# text_encoder.py (Version finale avec support des lots)

import subprocess
import time
import requests

def start_server(executable_path, model_path, port):
    """
    Lance le serveur llama.cpp et ATTEND qu'il soit réellement prêt à
    accepter des requêtes avant de continuer.
    Retourne l'objet du processus en cas de succès, sinon None.
    """
    command = [
        executable_path, "-m", model_path,
        "--embeddings", "--port", str(port)
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"Une erreur est survenue lors du lancement du processus serveur : {e}")
        return None

    print(f"Serveur en cours de démarrage sur le port {port}... Attente de la disponibilité...")
    
    max_wait_time = 30
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        try:
            response = requests.post(f"http://localhost:{port}/embedding", json={"input": ["test"]}, timeout=1)
            if response.status_code == 200:
                print("Serveur prêt.")
                return process
        except requests.exceptions.RequestException:
            time.sleep(1)
            
    print(f"Erreur : Le serveur n'a pas répondu après {max_wait_time} secondes. Arrêt.")
    process.terminate()
    process.wait()
    return None


def get_embedding(prompt, port):
    """
    Interroge le serveur pour obtenir un vecteur d'embedding pour UNE SEULE phrase.
    (Non utilisé par le pipeline principal, mais gardé pour le débogage)
    """
    try:
        response = requests.post(
            f"http://localhost:{port}/embedding",
            json={"input": [prompt]},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data[0]['embedding'][0]
    except requests.exceptions.RequestException as e:
        print(f"Erreur de communication avec le serveur : {e}")
        return None
    except (KeyError, IndexError):
        print("Erreur : La réponse du serveur n'a pas le format attendu.")
        return None

# =========================================================================
# ==================== NOUVELLE FONCTION POUR LES LOTS ====================
# =========================================================================
def get_batch_embeddings(list_of_prompts, port):
    """
    Interroge le serveur pour obtenir les embeddings pour une LISTE de phrases
    en une seule requête.
    """
    if not list_of_prompts:
        return []
        
    try:
        # La seule différence est ici : on passe la liste entière
        response = requests.post(
            f"http://localhost:{port}/embedding",
            json={"input": list_of_prompts},
            timeout=30 # On augmente le timeout car le traitement est plus long
        )
        response.raise_for_status()
        data = response.json()
        
        # On extrait chaque embedding de la liste de résultats
        embeddings = [result['embedding'][0] for result in data]
        return embeddings

    except requests.exceptions.RequestException as e:
        print(f"Erreur de communication avec le serveur : {e}")
        return None
    except (KeyError, IndexError):
        print("Erreur : La réponse du serveur n'a pas le format attendu.")
        return None
# =========================================================================
# ========================== FIN DE LA NOUVEAUTÉ ==========================
# =========================================================================