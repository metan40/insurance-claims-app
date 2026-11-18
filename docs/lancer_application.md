# Faire tourner Insurance Claims Assistance, version 19.5.0

Ce guide décrit les six façons d'utiliser l'application. Pour chacune, il
donne les commandes exactes, ce qui doit s'afficher et la manière de
vérifier que l'application est prête. Le programme est le même dans tous
les cas : le modèle d'embeddings `BAAI/bge-m3` rapproche la question des 201
documents, puis le modèle de langage `llama3.2:3b` rédige la réponse. Ce
modèle tourne sur la même machine, grâce au logiciel Ollama.

## 1. Ce qui est téléchargé et la mémoire nécessaire

Au premier lancement, une installation locale télécharge trois éléments. Ils
restent ensuite sur le disque : les lancements suivants ne les téléchargent
plus.

| Élément | Rôle | Taille |
|---|---|---|
| `llama3.2:3b` (Ollama) | rédige la réponse | 2,0 Go |
| `BAAI/bge-m3` (Hugging Face) | calcule les embeddings des questions et des documents | 2,3 Go |
| PyTorch pour processeur et dépendances Python | exécutent bge-m3 et l'API | environ 0,5 Go |

En fonctionnement, le modèle de langage occupe environ 2,5 Go de mémoire :
2,0 Go pour ses poids et le reste pour le cache de calcul d'un contexte de
2 048 tokens. bge-m3 occupe environ 2,3 Go. Il faut donc environ 5 Go de
mémoire libre, en plus de ce qu'utilise le système. La durée du premier
lancement dépend surtout du débit Internet, puisqu'il faut télécharger
4,8 Go. Elle n'a pas été mesurée sur un poste de travail.

Durées de réponse mesurées le 18/11/2026, sans carte graphique :

- sur le Space (8 processeurs virtuels) : de 37 à 85 secondes ;
- sur une machine GitHub (4 processeurs virtuels) : de 27 à 102 secondes.

## 2. Comment savoir que l'application est prête

Le serveur web démarre avant que les modèles aient fini de se charger. La
page d'état indique où en est ce chargement :

- en ligne : <https://metan40-insurance-claims-assistance.hf.space/ready>
- en local : <http://localhost:7860/ready>

Tant qu'un modèle n'est pas chargé, la page renvoie le code 503 et le champ
`checks` indique ce qui manque. L'application est prête quand la page
affiche ces trois valeurs :

```
"status": "ready"
"app_version": "19.5.0"
"retrieval_engine": "embeddings"
```

Si `retrieval_engine` vaut `hybrid`, l'application tourne avec l'ancienne
recherche TF-IDF de la version 19.4.0. Cela veut dire que le réglage
`RETRIEVAL_ENGINE` a été oublié (voir la section 5).

## 3. Façon A : utiliser l'application en ligne

L'application est publiée sur un Space Hugging Face : un serveur hébergé par
Hugging Face, qui exécute l'image Docker de l'application. Il n'y a rien à
installer, un navigateur suffit.

1. Ouvrir la page d'état
   <https://metan40-insurance-claims-assistance.hf.space/ready> et vérifier
   les trois valeurs de la section 2.
2. Ouvrir l'interface <https://metan40-insurance-claims-assistance.hf.space/>,
   taper une question et cliquer sur « Rechercher ».

Le Space tourne sur un matériel payant et sa mise en veille est désactivée :
il reste disponible même sans visite. À chaque mise à jour du code, il
reconstruit l'image et recharge les deux modèles. Pendant ce temps, la page
d'état renvoie 503.

Si la page affiche que le Space est en pause ou en erreur, son propriétaire
le relance depuis
<https://huggingface.co/spaces/metan40/insurance_claims_assistance> :
onglet Settings, bouton « Restart this Space ». En attendant, la façon F
donne une adresse publique à une installation locale.

## 4. Façon B : en local avec Docker

Docker exécute l'application dans un conteneur : un environnement isolé qui
contient Linux, Python, Ollama et le code. C'est la même image que sur le
Space, donc les réglages sont identiques sans rien configurer.

Prérequis : Docker installé et démarré. Sous Windows, c'est Docker Desktop,
dont l'icône doit être visible près de l'horloge.

Les commandes se tapent dans un terminal ouvert dans le dossier du dépôt,
celui qui contient `Dockerfile`, `README.md` et `VERSION`.

1. Construire l'image : `docker build -t insurance-claims .`
   L'option `-t insurance-claims` donne le nom `insurance-claims` à l'image.
   Le point final désigne le dossier courant, où se trouve le `Dockerfile`.
2. Créer le volume, une seule fois : `docker volume create donnees`
   Un volume est un espace disque géré par Docker, qui survit à l'arrêt du
   conteneur. Les modèles y sont rangés, ce qui évite de les retélécharger à
   chaque lancement.
3. Lancer le conteneur :
   `docker run -p 7860:7860 -v donnees:/data insurance-claims`
   - `-p 7860:7860` relie le port 7860 du poste au port 7860 du conteneur ;
   - `-v donnees:/data` place le volume `donnees` dans le dossier `/data` du
     conteneur.
4. Attendre que <http://localhost:7860/ready> soit prête (section 2), puis
   ouvrir <http://localhost:7860/>.
5. Pour arrêter l'application, appuyer sur `Ctrl + C` dans le terminal.

## 5. Façon C : en local sous Windows, sans Docker

Le script `scripts\lancer_local_windows.ps1` fait quatre choses, dans cet
ordre :

1. il vérifie qu'Ollama répond à l'adresse `http://127.0.0.1:11434`, et
   s'arrête avec un message si ce n'est pas le cas ;
2. il télécharge `llama3.2:3b` avec `ollama pull`, ce qui ne fait rien si le
   modèle est déjà présent ;
3. il crée un environnement Python isolé dans le dossier `.venv` et y
   installe les dépendances de `requirements.txt` ;
4. il applique les réglages du Space (`OLLAMA_MODEL=llama3.2:3b`,
   `RETRIEVAL_ENGINE=embeddings`, délai de réponse de 300 secondes), puis
   démarre l'application sur `127.0.0.1`, port 7860.

Prérequis : Python 3.10 ou plus récent, et Ollama pour Windows installé.

1. Démarrer Ollama depuis le menu Démarrer. Son icône apparaît près de
   l'horloge.
2. Ouvrir PowerShell et aller dans le dossier du dépôt :
   `cd $HOME\Desktop\insurance-claims-app-19.5.0`
3. Lancer le script :
   `powershell -ExecutionPolicy Bypass -File scripts\lancer_local_windows.ps1`
   L'option `-ExecutionPolicy Bypass` autorise ce script pour cette seule
   commande, sans modifier la sécurité de Windows.
4. Le terminal affiche les étapes 1/4 à 4/4, puis la ligne
   `Uvicorn running on http://127.0.0.1:7860`. Attendre que
   <http://localhost:7860/ready> soit prête, puis ouvrir
   <http://localhost:7860/>.
5. Pour arrêter l'application, appuyer sur `Ctrl + C` dans PowerShell.

## 6. Façon D : en local sous Linux ou macOS, sans Docker

Prérequis : Python 3.10 ou plus récent, et Ollama installé et lancé, soit
avec la commande `ollama serve`, soit avec l'application Ollama.

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
OLLAMA_MODEL=llama3.2:3b RETRIEVAL_ENGINE=embeddings uvicorn src.app:app --host 127.0.0.1 --port 7860
```

Les deux variables placées avant `uvicorn` ne s'appliquent qu'à cette
commande. Sans `RETRIEVAL_ENGINE=embeddings`, l'application utilise la
recherche de la version 19.4.0.

## 7. Les adresses en local

`localhost` est le nom de l'ordinateur sur lequel on travaille. Son adresse
IP est toujours `127.0.0.1` : <http://localhost:7860/> et
<http://127.0.0.1:7860/> ouvrent la même page.

`7860` est le numéro de port : l'entrée réseau sur laquelle l'application
attend les requêtes. Plusieurs programmes peuvent tourner sur le même
ordinateur, chacun sur son propre port. Ollama, par exemple, utilise le port
11434.

Le paramètre `--host` fixe qui peut joindre l'application :

- `--host 127.0.0.1` (façons C et D) : seul l'ordinateur lui-même ;
- `--host 0.0.0.0` (façon B, dans le conteneur) : toute machine qui peut
  joindre l'ordinateur par le réseau.

## 8. Façon E : depuis un autre appareil du même réseau

Cette façon sert à ouvrir l'application depuis un téléphone ou un second
ordinateur connecté au même Wi-Fi.

1. Lancer l'application avec `--host 0.0.0.0`. Pour la façon C, remplacer
   `127.0.0.1` par `0.0.0.0` à la dernière ligne du script. La façon B
   utilise déjà ce réglage.
2. Trouver l'adresse IP du poste qui fait tourner l'application. Sous
   Windows, taper `ipconfig` et lire la ligne « Adresse IPv4 » de la carte
   Wi-Fi, par exemple `192.168.1.25`. Sous Linux, taper `ip addr`.
3. Sur l'autre appareil, ouvrir `http://192.168.1.25:7860/`, avec l'adresse
   trouvée à l'étape 2.
4. Si Windows demande d'autoriser Python dans le pare-feu, cocher seulement
   « Réseaux privés ». Avec « Réseaux publics », l'application serait aussi
   visible sur les Wi-Fi publics.

## 9. Façon F : depuis Internet, en secours

Cloudflare Tunnel ouvre une connexion sortante du poste vers les serveurs de
Cloudflare. Ces serveurs donnent une adresse publique
`https://…trycloudflare.com` et transmettent au poste les requêtes reçues à
cette adresse. Aucun port n'est ouvert sur la box, et aucun compte n'est
nécessaire.

1. Installer le programme, une seule fois. Sous Windows :
   `winget install Cloudflare.cloudflared`
2. Laisser l'application tourner (façon B, C ou D), ouvrir un second
   terminal et taper : `cloudflared tunnel --url http://localhost:7860`
3. Le terminal affiche une adresse de la forme
   `https://mots-au-hasard.trycloudflare.com`. Elle donne accès aux mêmes
   pages : `/`, `/claims`, `/ready`, `/docs`.
4. Cette adresse change à chaque lancement de `cloudflared`. Elle cesse de
   fonctionner dès que le poste est éteint ou que le terminal est fermé.
   C'est donc une solution de secours, pas une adresse à envoyer à l'avance.

## 10. Vérifier une installation par un script

Le script `evaluation/verifier_application.py` interroge l'application comme
le ferait un navigateur. Il n'utilise que la bibliothèque standard de Python :

```bash
python evaluation/verifier_application.py --url http://localhost:7860 --repetitions 1
```

Il attend que la page d'état soit prête, puis pose les dix questions du
fichier `evaluation/questions_jury.csv`. Pour chaque question, il affiche le
statut, le moteur de recherche, la source citée, le nombre de phrases
masquées et la durée. Pour vérifier une autre installation, on remplace
l'adresse par celle du Space ou par celle donnée par `cloudflared`.

## 11. Problèmes connus

| Ce qui s'affiche | Cause | Que faire |
|---|---|---|
| La page d'état renvoie 503 et `checks` contient une valeur `false` | un modèle est en cours de téléchargement ou de chargement | attendre, puis recharger la page ; le champ à `false` indique quel modèle manque |
| Le script Windows affiche « Ollama ne répond pas » | Ollama n'est pas démarré | démarrer Ollama, puis relancer le script |
| `retrieval_engine` vaut `hybrid` | `RETRIEVAL_ENGINE=embeddings` n'a pas été fixé | utiliser la façon B ou C, ou ajouter le réglage (façon D) |
| `Address already in use` ou « port 7860 déjà utilisé » | une autre instance de l'application tourne déjà | arrêter l'autre instance avec `Ctrl + C`, ou changer de port : `--port 7861` (C, D) ou `-p 7861:7860` (B), puis ouvrir <http://localhost:7861/> |
| La réponse dépasse le délai | le modèle de langage manque de processeur ou de mémoire | fermer les autres programmes et vérifier qu'environ 5 Go de mémoire sont libres (section 1) |
| Le Space est en pause ou en erreur | arrêt du matériel ou problème de facturation | relancer le Space (section 3) et utiliser la façon F en attendant |

## 12. Récapitulatif des commandes

| But | Commande |
|---|---|
| Aller dans le dossier du dépôt (Windows) | `cd $HOME\Desktop\insurance-claims-app-19.5.0` |
| Lancer sous Windows sans Docker (C) | `powershell -ExecutionPolicy Bypass -File scripts\lancer_local_windows.ps1` |
| Construire l'image Docker (B) | `docker build -t insurance-claims .` |
| Créer le volume des modèles (B, une fois) | `docker volume create donnees` |
| Lancer avec Docker (B) | `docker run -p 7860:7860 -v donnees:/data insurance-claims` |
| Arrêter l'application | `Ctrl + C` dans le terminal où elle tourne |
| Télécharger le modèle de langage | `ollama pull llama3.2:3b` |
| Voir les modèles Ollama installés | `ollama list` |
| Lancer sans Docker sous Linux ou macOS (D) | `OLLAMA_MODEL=llama3.2:3b RETRIEVAL_ENGINE=embeddings uvicorn src.app:app --port 7860` |
| Trouver l'adresse IP du poste (E) | `ipconfig` (Windows) ou `ip addr` (Linux) |
| Installer cloudflared sous Windows (F) | `winget install Cloudflare.cloudflared` |
| Ouvrir l'accès Internet de secours (F) | `cloudflared tunnel --url http://localhost:7860` |
| Vérifier une installation | `python evaluation/verifier_application.py --url http://localhost:7860 --repetitions 1` |
| Lancer les tests automatisés | `pip install -r requirements-dev.txt`, puis `python -m pytest -q` |

## 13. Adresses complètes

| Page | En ligne (Space) | En local |
|---|---|---|
| Interface de l'assistant documentaire | <https://metan40-insurance-claims-assistance.hf.space/> | <http://localhost:7860/> |
| Tableau de bord des sinistres fictifs | <https://metan40-insurance-claims-assistance.hf.space/claims> | <http://localhost:7860/claims> |
| État de l'application | <https://metan40-insurance-claims-assistance.hf.space/ready> | <http://localhost:7860/ready> |
| Version et modèle configuré | <https://metan40-insurance-claims-assistance.hf.space/version> | <http://localhost:7860/version> |
| Réponse minimale de l'API | <https://metan40-insurance-claims-assistance.hf.space/health> | <http://localhost:7860/health> |
| Documentation interactive de l'API | <https://metan40-insurance-claims-assistance.hf.space/docs> | <http://localhost:7860/docs> |
| Aide à la décision du dossier C00020 | <https://metan40-insurance-claims-assistance.hf.space/api/claims/C00020/decision-support> | <http://localhost:7860/api/claims/C00020/decision-support> |
| Réglages et redémarrage du Space | <https://huggingface.co/spaces/metan40/insurance_claims_assistance> | sans objet |
| Code source | <https://github.com/metan40/insurance-claims-app> | sans objet |

Depuis un autre appareil du réseau (façon E), on remplace `localhost` par
l'adresse IP du poste, par exemple <http://192.168.1.25:7860/claims>. Depuis
Internet (façon F), on remplace le début de l'adresse par celle que donne
`cloudflared`, par exemple `https://mots-au-hasard.trycloudflare.com/claims`.
