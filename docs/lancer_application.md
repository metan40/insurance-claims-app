# Faire tourner Insurance Claims Assistance (19.5.0)

L'application est publiée à une adresse fixe, et elle peut aussi tourner sur
n'importe quel ordinateur. La version est la même partout : recherche par
embeddings bge-m3, modèle `llama3.2:3b` exécuté par Ollama.

| Façon | Pour qui | Il faut | Temps du premier lancement |
|---|---|---|---|
| A. En ligne (Space) | tout le monde, depuis un navigateur | rien | aucun |
| B. En local avec Docker | un poste avec Docker | Docker, environ 8 Go de mémoire libre | 10 à 20 min (téléchargements) |
| C. En local sous Windows, sans Docker | un PC Windows | Python 3.10+, Ollama pour Windows | 10 à 20 min |
| D. En local sous Linux ou macOS, sans Docker | un poste de développement | Python 3.10+, Ollama | 10 à 20 min |
| E. Depuis un autre appareil du même réseau | téléphone, autre PC | B, C ou D lancé | immédiat |
| F. Depuis Internet, en secours | n'importe qui | B, C ou D lancé, `cloudflared` | 1 min |

Dans tous les cas, l'application est prête quand l'adresse `…/ready`
répond `"status": "ready"`.

## A. En ligne

- Application : <https://metan40-insurance-claims-assistance.hf.space/>
- Contrôle : <https://metan40-insurance-claims-assistance.hf.space/ready>,
  qui doit afficher `ready`, la version `19.5.0` et `retrieval_engine: embeddings`.
- Le Space reste allumé en permanence (matériel payant, mise en veille
  désactivée). Après une mise à jour, il redémarre et recharge les modèles :
  il faut quelques minutes.
- Si la page indique que le Space est en pause ou arrêté : le propriétaire le
  relance dans Settings → *Restart this Space* sur Hugging Face. Pendant ce
  temps, passer à F.

## B. En local avec Docker (identique au Space)

Dans le dossier du dépôt :

```bash
docker build -t insurance-claims .
docker volume create donnees
docker run -p 7860:7860 -v donnees:/data insurance-claims
```

Ouvrir ensuite <http://localhost:7860>. Le volume `donnees` garde les
modèles : seul le premier démarrage les télécharge.

## C. En local sous Windows, sans Docker

1. Lancer l'application **Ollama** (son icône apparaît près de l'horloge).
2. Dans **PowerShell**, depuis le dossier du dépôt :

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\lancer_local_windows.ps1
   ```

   Le script vérifie qu'Ollama répond, télécharge `llama3.2:3b`, crée un
   environnement Python `.venv`, installe les dépendances, puis démarre
   l'application avec les réglages du Space.
3. Ouvrir <http://localhost:7860>.

## D. En local sous Linux ou macOS, sans Docker

Avec Ollama lancé :

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
OLLAMA_MODEL=llama3.2:3b RETRIEVAL_ENGINE=embeddings uvicorn src.app:app --host 127.0.0.1 --port 7860
```

Sans `RETRIEVAL_ENGINE=embeddings`, l'application utilise l'ancienne
recherche hybride TF-IDF de la version 19.4.0.

## Adresses en local

- `localhost` désigne l'ordinateur lui-même. Son adresse IP est `127.0.0.1` :
  <http://localhost:7860> et <http://127.0.0.1:7860> sont équivalents.
- `7860` est le port, c'est-à-dire la « porte » sur laquelle l'application
  écoute.
- Lancée avec `--host 127.0.0.1`, l'application n'est visible que depuis cet
  ordinateur. C'est le réglage le plus sûr.

## E. Depuis un autre appareil du même réseau

1. Lancer l'application avec `--host 0.0.0.0` au lieu de `127.0.0.1`. Avec
   Docker (B), c'est déjà le cas.
2. Trouver l'adresse IP du poste : `ipconfig` sous Windows (ligne « Adresse
   IPv4 »), `ip addr` sous Linux. Par exemple `192.168.1.25`.
3. Sur l'autre appareil, ouvrir `http://192.168.1.25:7860`.
4. Si le pare-feu le demande, autoriser l'accès sur les réseaux privés
   seulement.

## F. Depuis Internet, en secours (Cloudflare Tunnel)

Avec l'application lancée en B, C ou D, dans un second terminal :

```bash
cloudflared tunnel --url http://localhost:7860
```

La commande affiche une adresse `https://….trycloudflare.com`. N'importe qui
peut l'ouvrir tant que le poste reste allumé. C'est gratuit, sans compte, et
aucun port n'est ouvert sur la box. Installation : `winget install
Cloudflare.cloudflared` sous Windows, ou via le site de Cloudflare.
L'adresse change à chaque lancement : la communiquer au moment de la
démonstration.

## Vérifier que tout fonctionne, quelle que soit la façon

```bash
python evaluation/verifier_application.py --url http://localhost:7860 --repetitions 1
```

Remplacer l'adresse par celle du Space ou du tunnel. Le script attend
`/ready`, pose les questions de démonstration et affiche, pour chacune, le
statut, la source citée et la durée.

## En cas de problème

| Symptôme | Cause probable | Que faire |
|---|---|---|
| `/ready` répond 503 | un modèle est encore en cours de téléchargement ou de chargement | attendre quelques minutes, puis recharger la page |
| « Ollama ne répond pas » | Ollama n'est pas lancé | lancer Ollama, puis relancer |
| `retrieval_engine: hybrid` | `RETRIEVAL_ENGINE=embeddings` manquant | relancer avec ce réglage (D) ou utiliser B ou C |
| réponse très lente ou délai dépassé | poste peu puissant ou mémoire insuffisante | fermer les autres programmes ; prévoir environ 8 Go libres |
| le port 7860 est occupé | une autre instance tourne déjà | fermer l'autre instance, ou changer de port (`--port 7861`, `-p 7861:7860`) |
| le Space est en pause | matériel ou facturation | relancer le Space (A) ; en attendant, utiliser F |

## Récapitulatif des commandes

| But | Commande |
|---|---|
| Aller dans le dossier (Windows) | `cd $HOME\Desktop\insurance-claims-app-19.5.0` |
| Lancer sous Windows (C) | `powershell -ExecutionPolicy Bypass -File scripts\lancer_local_windows.ps1` |
| Construire l'image Docker (B) | `docker build -t insurance-claims .` |
| Créer le volume des modèles (B, une fois) | `docker volume create donnees` |
| Lancer avec Docker (B) | `docker run -p 7860:7860 -v donnees:/data insurance-claims` |
| Arrêter l'application | `Ctrl + C` dans le terminal où elle tourne |
| Télécharger le modèle de langage | `ollama pull llama3.2:3b` |
| Voir les modèles Ollama installés | `ollama list` |
| Lancer sans Docker (D) | `OLLAMA_MODEL=llama3.2:3b RETRIEVAL_ENGINE=embeddings uvicorn src.app:app --port 7860` |
| Trouver l'adresse IP du poste (E) | `ipconfig` (Windows), `ip addr` (Linux) |
| Installer cloudflared (Windows) | `winget install Cloudflare.cloudflared` |
| Ouvrir l'accès Internet de secours (F) | `cloudflared tunnel --url http://localhost:7860` |
| Vérifier une installation | `python evaluation/verifier_application.py --url http://localhost:7860 --repetitions 1` |
| Lancer les tests automatisés | `pip install -r requirements-dev.txt` puis `python -m pytest -q` |

| Adresse | Rôle |
|---|---|
| `…/` | interface (assistant documentaire) |
| `…/claims` | tableau de bord des sinistres fictifs |
| `…/ready` | état de l'application (prête ou non) |
| `…/version` | version et modèle |
| `…/docs` | documentation interactive de l'API |
