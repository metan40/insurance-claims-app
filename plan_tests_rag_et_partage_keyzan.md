# Plan de test de l'application RAG — préparation des essais avec Keyzan

## Objectif

Je veux vérifier le comportement de l'application complète, depuis la question saisie dans l'interface jusqu'à la réponse générée par le LLM. Les scores de recherche documentaire ne suffisent pas à confirmer que la réponse finale est exacte et fidèle aux sources.

Ce document décrit les essais à effectuer. Il ne signifie pas que les tests ont déjà tous réussi ni que l'application est déjà déployée.

## Architecture actuelle à tester

1. L'interface web est servie par la route `GET /` et le fichier `static/index.html`.
2. L'API reçoit les questions sur `POST /search`.
3. L'application encode la question avec `paraphrase-multilingual-MiniLM-L12-v2`.
4. La recherche sémantique sélectionne jusqu'à 20 documents candidats.
5. Le modèle `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` réordonne les candidats.
6. L'application envoie les cinq premiers documents à Ollama, avec le modèle `llama3.2:3b`.
7. L'API retourne la réponse générée ainsi que les documents retenus.

Le code actuel appelle Ollama à l'adresse `http://localhost:11434/api/generate`. Avant de donner un accès à distance à Keyzan, je dois donc vérifier où l'application et Ollama seront hébergés. `localhost` désigne toujours la machine sur laquelle le processus de l'application s'exécute.

## Préparation de l'environnement

- Installer une version compatible de Python et les dépendances du projet depuis `requirements.txt`.
- Installer Ollama sur la machine qui hébergera le modèle, puis télécharger le modèle avec `ollama pull llama3.2:3b`.
- Vérifier que les fichiers `Data/insurance-fr.csv` et `static/index.html` sont présents.
- Pour plus de simplicité, lancer les commandes depuis le répertoire racine. Les chemins du CSV et de la page HTML sont construits à partir de l’emplacement de `src/app.py` ; le scénario de validation doit aussi confirmer que ces fichiers restent accessibles lorsque le répertoire de travail est différent.
- Pour un test distant, configurer l'accès réseau de manière contrôlée. Je ne dois pas exposer directement un service de test sans avoir défini qui peut y accéder.

## Démarrage local sous Windows PowerShell

Depuis le dossier racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2:3b
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Après le démarrage, j'ouvre `http://127.0.0.1:8000/` dans le navigateur. Je peux vérifier le contrat de l'API avec une requête `POST /search` contenant un JSON de la forme `{"question":"Quelles sont les conditions de prise en charge ?"}`. Cette question sert d'exemple de format ; le contenu de la réponse doit être vérifié par rapport au corpus.

## Scénarios à exécuter

| ID | Essai | Résultat attendu | Résultat réel / notes |
|---|---|---|---|
| T01 | Lancer l'API | Le serveur démarre et les modèles se chargent sans exception. | À compléter après exécution |
| T02 | Ouvrir la page d'accueil | L'interface est affichée et permet de saisir une question. | À compléter après exécution |
| T03 | Poser une question couverte par le corpus | L'API renvoie une réponse et des documents récupérés. Je vérifie la réponse manuellement. | À compléter après exécution |
| T04 | Poser une question vide | L'application répond de manière contrôlée sans lancer une génération inutile. | À compléter après exécution |
| T05 | Poser une question hors corpus | Le modèle reconnaît que les documents ne suffisent pas et n'invente pas de réponse. | À compléter après exécution |
| T06 | Vérifier les sources | Les titres et extraits renvoyés correspondent bien aux documents utilisés pour la réponse. | À compléter après exécution |
| T07 | Arrêter Ollama pendant un essai | L'application signale clairement l'échec technique. | À compléter après exécution |
| T08 | Mesurer la latence | Je note le temps total et les éventuels dépassements de délai. | À compléter après exécution |
| T09 | Tester depuis un autre appareil | La page et l'API sont accessibles uniquement par le canal de test prévu. | À compléter après configuration du déploiement |
| T10 | Faire tester par Keyzan | Keyzan note la question, la réponse, les sources et les problèmes observés. | À planifier |

## Fiche de retour pour Keyzan

Pour chaque essai, je demande à Keyzan de noter :

- la question saisie ;
- la réponse affichée par l'application ;
- les sources présentées ;
- si la réponse est correcte, partielle, non justifiée ou hors sujet ;
- si l'interface est facile à utiliser ;
- le temps d'attente approximatif ;
- une capture d'écran ou le message d'erreur lorsqu'un problème apparaît.

Je conserverai les retours dans un tableau de suivi et je corrigerai les problèmes reproductibles avant d'élargir les essais.

## Points à régler avant un partage distant

- L'adresse Ollama est codée en dur dans `src/app.py`. Je dois la rendre configurable ou installer Ollama sur le même hôte que l'application.
- Je dois vérifier les dépendances et la disponibilité des modèles sur l'environnement de déploiement.
- Je dois prévoir un accès limité et contrôlé, plutôt que d'exposer publiquement l'API sans protection.
- Je dois vérifier le comportement des erreurs et des délais d'attente, notamment lorsque le modèle ne répond pas.
- Je dois éviter d'utiliser des données personnelles réelles pour les premiers tests partagés.

## Critère de validation

Je considérerai la version prête pour les essais de Keyzan lorsque les scénarios essentiels T01 à T08 auront été exécutés, que les erreurs bloquantes auront été corrigées et qu'un accès de test contrôlé aura été préparé. Je noterai les résultats réels des tests au lieu de déclarer les scénarios réussis à l'avance.

## Étape complémentaire — validation des entrées et erreurs Ollama

J’ai ajouté une première couche de robustesse à l’API sans modifier le modèle d’embeddings, le classement initial, le reranker ni la sélection des cinq documents.

- L’API accepte une question non vide après suppression des espaces de début et de fin, avec une longueur maximale de 2 000 caractères.
- FastAPI renvoie une erreur HTTP 422 lorsque la question est vide ou dépasse la limite.
- Si Ollama ne répond pas, renvoie un JSON invalide ou ne fournit pas de texte exploitable, l’API renvoie une erreur HTTP 502 plutôt que de présenter le message technique comme une réponse d’assurance.
- Les détails techniques restent dans les logs du serveur, tandis que le client reçoit un message d’erreur plus générique.

**Hypothèse à vérifier :** ces contrôles doivent rendre les erreurs d’entrée et d’indisponibilité plus faciles à distinguer d’une réponse normale.

**État des tests :** cette modification a été vérifiée par compilation syntaxique. Les tests d’intégration nécessitant le lancement de l’application, le téléchargement/chargement des modèles et un service Ollama actif restent à exécuter dans l’environnement du projet ; aucune réponse générée par Ollama n’est revendiquée comme testée ici.



## Complément version 5 — contrôle des citations

La version 5 expose un champ `citation_audit` qui vérifie les identifiants au format `[S<number>]`. Dans la campagne de validation, je dois contrôler trois cas distincts : une citation correspondant à une source disponible, un identifiant inconnu et une réponse sans citation. Je dois ensuite vérifier manuellement si le contenu de la source justifie réellement l'affirmation : le contrôle automatisé ne juge pas la fidélité sémantique.

Je dois aussi lancer l'application depuis un répertoire différent de la racine du projet pour confirmer que les chemins vers le CSV et la page d'accueil restent valides. Les tests unitaires existants sont une étape de vérification du code ; ils ne remplacent pas les essais avec les modèles réels et Ollama actif.

## Vérifications complémentaires introduites en V6

- Vérifier que l'interface distingue les identifiants reconnus, inconnus et les sources disponibles mais non citées.
- Vérifier que les libellés précisent que le score brut du reranker n'est pas une probabilité et que la similarité cosinus ne prouve pas la véracité d'une réponse.
- Vérifier que `score_explanation` est présent dans la réponse JSON de `/search`.


## Vérifications complémentaires introduites en V7

- Vérifier que la réponse JSON de `/search` contient `pipeline_metrics` avec les compteurs de documents, candidats et résultats renvoyés.
- Vérifier automatiquement que les chemins vers le CSV et la page HTML restent valides lorsque le répertoire de travail n’est pas la racine du projet.
- Vérifier que les durées de l'embedding, de la recherche, du reranking, de la génération et de la requête totale sont numériques et non négatives.
- Pendant les essais avec les modèles réels, relever les durées sur plusieurs questions et documenter la machine, le modèle chargé et les conditions de l'essai. Les durées des tests simulés ne doivent pas être interprétées comme des performances réelles.


## Résultats des premiers essais de bout en bout — V7

Ces résultats proviennent de deux exécutions réalisées dans l'interface avec le serveur local et le modèle Ollama actif. Ils ne constituent pas une validation exhaustive.

| Essai | Observation réelle | Conclusion provisoire |
|---|---|---|
| Question générale sur la prise en charge d'un sinistre | La réponse cite S1, S2, S3 et S5. Certaines exclusions présentées sous S2 semblent provenir du contenu de S4, relatif à une garantie de revente personnelle ou familiale. Durée totale observée : 61,63 s. | Recherche fonctionnelle, mais réponse trop large et attribution des sources à vérifier. |
| Question ciblée sur les exclusions de la garantie de revente personnelle ou familiale | Les documents S1 et S3 contiennent les exclusions pertinentes. La réponse reprend la liste de S3 mais ne comporte aucune citation ; l'interface le signale. Durée totale observée : 64,99 s. | Pertinence du document satisfaisante pour cet exemple ; défaut de citation constaté. |

Les durées observées dans ces deux essais sont des mesures ponctuelles sur la machine de test. La génération a pris respectivement environ 57,53 s et 61,29 s ; ces valeurs ne doivent pas être généralisées.

## Correctif V7.1 à revalider

Le code a été modifié pour demander une seule révision lorsque les citations sont absentes ou inconnues, puis afficher un message prudent si cette révision échoue également. Les 22 tests automatisés passent dans l'environnement de vérification. Cette modification n'a pas encore été revalidée avec Ollama réel. Il faut répéter la question ciblée et contrôler : (1) que la réponse révisée contient une référence reconnue, ou que le message de repli apparaît ; (2) que les références soutiennent effectivement les affirmations ; (3) la nouvelle durée de génération si un second appel est effectué.
