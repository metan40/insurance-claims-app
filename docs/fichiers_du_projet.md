# Fichiers du projet Insurance Claims Assistance, version 19.5.0

Ce document décrit chaque fichier du dépôt. Pour chacun, il indique son
rôle et, pour les données, leur origine. Il complète et met à jour le
tableau 2 du rapport v19, « Composants et responsabilités ». Ce tableau
date de la version 19 et ne contient donc pas les fichiers ajoutés depuis :
la recherche par embeddings, les mesures, la vérification locale et le
script Windows. Le dossier `experiments/` contient les scripts des versions
précédentes, conservés comme archives. Il ne sert pas à l'application en
service et n'est pas détaillé ici.

## 1. Données (`Data/`)

| Fichier | Contenu | Origine |
|---|---|---|
| `Data/insurance-fr.csv` | 201 documents d'information sur l'assurance habitation, en français, sur deux colonnes : `title` (titre, souvent formulé comme une question) et `content` (texte). C'est le corpus de l'assistant documentaire. | Jeu de données public `zelros/insurance-fr`, publié sur Hugging Face par la société Zelros (dernière mise à jour le 21/10/2023). Il a été téléchargé par le script `experiments/scripts/download_dataset.py` depuis `https://huggingface.co/datasets/zelros/insurance-fr/resolve/main/data.csv`. La fiche du jeu de données n'indique ni l'origine des textes ni de licence ; l'historique du jeu montre des mises à jour par un membre de Zelros en octobre 2023. Une vérification par échantillon (18/11/2026) établit une origine partielle : 66 documents sur 201 citent l'assureur MMA (groupe Covéa) et reprennent ses noms de garanties (« renfort installations extérieures », « renfort valeur à neuf ») ; plusieurs titres correspondent à des guides en ligne de l'assureur Luko (« L'attestation télétravail de l'assurance habitation », assurance d'une location saisonnière). Les droits sur ces textes appartiennent donc vraisemblablement à leurs éditeurs : le corpus ne doit servir qu'à une démonstration non commerciale ; un usage commercial exige des documents dont les droits sont établis. |
| `Data/synthetic_claims.csv` | 120 dossiers de sinistres : type, garantie, montant demandé, ancienneté du client, sinistres précédents, canal de déclaration, dossier complet, statut (72 acceptés, 27 refusés, 21 en attente), date de déclaration, délai de traitement. Ce fichier alimente le tableau de bord `/claims`. | Données fictives créées pour la démonstration, sans source externe. Leur conception est décrite dans le document de cadrage `docs/cadrage/Projet python.docx` : « avant de coder, nous allons concevoir le jeu de données fictif […] une cinquantaine ou une centaine de sinistres fictifs avec des variables réalistes, des Pending, des acceptés, des refusés et quelques cas volontairement ambigus ». Le fichier a été ajouté en version 7.5.0 (`docs/historique/changelogs/CHANGELOG_v10.md`). Chaque ligne porte la mention `FICTIF_DEMONSTRATION` dans la colonne `jeu_de_donnees`. Le tableau 3 du rapport v19 décrit chaque variable. |

## 1 bis. Modèles utilisés et leur origine

Les deux modèles ne sont pas dans le dépôt. Ils sont téléchargés au premier
lancement, depuis la source indiquée ci-dessous.

| Modèle | Rôle dans l'application | Origine | Licence |
|---|---|---|---|
| `BAAI/bge-m3` | Transforme la question et chaque document en un vecteur de 1 024 nombres. Le document dont le vecteur est le plus proche de celui de la question (similarité cosinus) est le plus pertinent. | Publié sur Hugging Face (<https://huggingface.co/BAAI/bge-m3>) par le BAAI (Beijing Academy of Artificial Intelligence), un institut public de recherche en intelligence artificielle de Pékin. Article de référence : J. Chen et al., « BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation », 2024, arXiv:2402.03216. | MIT |
| `llama3.2:3b` | Rédige la réponse à partir des documents retrouvés, au format JSON imposé. | Modèle Llama 3.2 3B Instruct de Meta, distribué par la bibliothèque de modèles d'Ollama (<https://ollama.com/library/llama3.2>) dans une version compressée de 2,0 Go. | Llama 3.2 Community License (Meta) |

**Ce que veut dire le nom bge-m3.** BGE signifie *BAAI General Embedding*.
Les trois « M » désignent trois propriétés décrites dans l'article :
*multi-lingual* (plus de 100 langues, dont le français),
*multi-functionality* (le même modèle sait produire trois représentations :
un vecteur dense, des poids par mot comparables à TF-IDF, et un vecteur par
token) et *multi-granularity* (des textes de quelques mots jusqu'à
8 192 tokens). L'application n'utilise que le vecteur dense, de dimension
1 024, par la bibliothèque `sentence-transformers`. Le modèle compte environ
568 millions de paramètres, d'où ses 2,3 Go en précision de 32 bits.

**Le document qui a justifié l'abandon de TF-IDF.** Ce document est une
mesure faite dans ce projet. Le script `evaluation/experience_embeddings.py`,
exécuté par le workflow `experience-embeddings.yml`, a comparé les moteurs
de recherche, d'abord sur un jeu de développement, puis une seule fois sur
les 30 questions reformulées tenues à part (`evaluation/heldout_questions.csv`).
Le bon document se trouvait parmi les 3 premiers dans 13 cas sur 30 avec la
recherche hybride TF-IDF de la version 19.4.0, et dans 26 cas sur 30 avec
bge-m3 seul (87 %, intervalle de Wilson à 95 % [70 % ; 95 %]). Le test de
McNemar exact donne p = 0,001 : 14 questions gagnées contre 1 perdue. La
fusion de bge-m3 avec TF-IDF faisait moins bien que bge-m3 seul (70 %). Les
résultats sont consignés dans `CHANGELOG.md` (version 19.5.0) et dans
`docs/decisions_mesurees.md` (décision 1). TF-IDF n'a pas disparu pour autant :
il reste le moteur de repli si bge-m3 ne peut pas être chargé.

## 2. Code de l'application (`src/`)

| Fichier | Rôle |
|---|---|
| `src/app.py` | L'API FastAPI. Elle déclare les routes (`/`, `/search`, `/ready`, `/version`, `/health`, `/claims`, `/api/claims/...`), lit les réglages (modèle, moteur de recherche, délais), appelle Ollama et mesure les durées. |
| `src/question_parsing.py` | Analyse la question avant la recherche : elle extrait les termes significatifs et applique le dictionnaire de synonymes métier (par exemple « nounou » devient « baby-sitter »). |
| `src/embedding_retrieval.py` | Recherche par embeddings, le moteur de la version 19.5.0. Elle charge `BAAI/bge-m3`, calcule une fois les vecteurs des 201 documents (mis en cache sur disque), puis classe les documents par similarité cosinus avec la question. |
| `src/retrieval.py` | Recherche hybride TF-IDF et lexicale, avec fusion des rangs (RRF). C'était le moteur de la version 19.4.0 ; c'est aujourd'hui le moteur de repli si bge-m3 n'est pas disponible. |
| `src/generation.py` | Génération typée de la réponse. Il construit le prompt, demande à Ollama une réponse JSON conforme à un schéma, vérifie que chaque extrait cité figure dans sa source, décide du statut affiché et applique la boucle « meilleure source d'abord, trois sources si nécessaire ». |
| `src/grounding.py` | Contrôle d'ancrage. Chaque phrase de la réponse doit être appuyée par les sources citées ; une phrase sans appui est masquée, et si plus rien n'est prouvé, l'application n'affiche que les extraits vérifiés. |
| `src/definitions.py` | Quand une source renvoie à une définition (« voir définition »), le document de définition est joint à la source. |
| `src/claims_analytics.py` | Tableau de bord des sinistres : indicateurs (taux d'acceptation sur les dossiers clos), entraînement et évaluation de la régression logistique, probabilité et contributions de chaque facteur pour un dossier. |
| `src/claim_decision.py` | Aide à la décision sur un dossier. Il réunit le score expliqué, les règles métier (points bloquants, alertes), les clauses liées au type de sinistre et une orientation pour le gestionnaire. |
| `src/__init__.py` | Fichier vide qui fait de `src` un paquet Python importable. |

## 3. Interface (`static/`)

| Fichier | Rôle |
|---|---|
| `static/index.html` | Page de l'assistant documentaire : saisie de la question, statut, réponse, extraits vérifiés, sources, similarité, durées. |
| `static/claims.html` | Page du tableau de bord : indicateurs, liste des dossiers, analyse et aide à la décision d'un dossier. |

## 4. Démarrage et déploiement

| Fichier | Rôle |
|---|---|
| `Dockerfile` | Recette de l'image Docker utilisée par le Space et par le lancement local avec Docker. Elle installe Python, Ollama et les dépendances, crée l'utilisateur 1000 exigé par Hugging Face et fixe les réglages (`OLLAMA_MODEL=llama3.2:3b`, `RETRIEVAL_ENGINE=embeddings`). |
| `docker-entrypoint.sh` | Script exécuté au démarrage du conteneur. Il démarre Ollama, choisit le dossier de stockage des modèles (`/data` s'il existe), télécharge et précharge `llama3.2:3b` en arrière-plan, puis lance l'API. |
| `scripts/lancer_local_windows.ps1` | Lancement local sous Windows sans Docker : il vérifie Ollama, télécharge le modèle, crée l'environnement Python `.venv`, installe les dépendances et démarre l'application avec les réglages du Space. |
| `.dockerignore` | Liste des fichiers exclus de l'image Docker (historique Git, environnements Python, caches). |
| `.env.example` | Exemple des variables de configuration, à recopier et adapter. |
| `requirements.txt` | Dépendances Python nécessaires à l'application (FastAPI, scikit-learn, PyTorch pour processeur, sentence-transformers, etc.). |
| `requirements-dev.txt` | Dépendances de l'application, plus pytest pour les tests. |
| `requirements-experiments.txt` | Dépendances des scripts archivés de `experiments/`. |
| `VERSION` | Numéro de version unique (`19.5.0`), lu par l'API et affiché par `/version` et `/ready`. |

## 5. Automatisation GitHub (`.github/workflows/`)

| Fichier | Rôle |
|---|---|
| `sync-to-hub.yml` | À chaque envoi sur GitHub, il lance les tests. Pour la branche `main` seulement, et seulement si les tests réussissent, il met à jour le Space Hugging Face. |
| `verification-locale.yml` | Construit l'image Docker sur une machine vierge, la lance avec un volume et pose les questions de démonstration. C'est la preuve que l'application fonctionne sans le serveur distant. |
| `verifier-space.yml` | Interroge l'application en ligne sans la redémarrer : il attend la version attendue sur `/ready`, puis pose les questions de démonstration. |
| `demonstration.yml` | Pose les cinq questions de démonstration au Space et interroge le tableau de bord ; les réponses complètes sont écrites dans le journal. |
| `evaluation-modeles.yml` | Mesure de bout en bout : il installe Ollama et un modèle, puis mesure les statuts et la conformité des réponses sur les jeux de questions. |
| `experience-embeddings.yml` | Expérience qui a comparé les moteurs de recherche (TF-IDF, bge-m3, fusion) sur les jeux dev et tenu à part. |
| `mesure-justification.yml` | Premier essai du contrôle par inférence textuelle (prémisses : extraits cités). |
| `analyse-justification-documents.yml` | Second essai du contrôle par inférence textuelle (prémisses : documents cités). |
| `comparer-versions.yml` | Comparaison appariée, question par question, de la version 19.4.0 et de la version 19.5.0. |
| `relancer-space.yml` | Désactive la mise en veille du Space, le redémarre et attend qu'il soit prêt. Ce workflow ne se lance que manuellement. |

## 6. Mesures (`evaluation/`)

| Fichier | Rôle |
|---|---|
| `questions_jury.csv` | Dix questions de démonstration, chacune avec son comportement attendu (complète, partielle, absente) et sa source attendue. |
| `heldout_questions.csv` | 30 questions reformulées, tenues à part, avec le document attendu. C'est le jeu de test de la recherche. |
| `paraphrase_questions.csv` | 40 questions reformulées utilisées pour le développement de la recherche. |
| `reponses_reference.csv` | Réponses de référence et faits obligatoires, qui servent à juger si une réponse est conforme. |
| `questions_hors_documents.csv` | 30 questions dont la réponse n'est pas dans le corpus, réparties à l'avance entre dev et test. |
| `results/evaluation_par_question.csv`, `results/evaluation_synthese.json` | Résultats de la mesure de la recherche (Recall@k, MRR) par question et en synthèse. |
| `evaluate_pipeline.py` | Mesure de la recherche : Recall@k et MRR. |
| `experience_embeddings.py` | Expérience comparant les moteurs de recherche. |
| `mesurer_statuts.py` | Mesure de bout en bout des statuts et de la conformité des réponses, avec enregistrement de chaque passage. |
| `comparer_modeles.py` | Tableau comparatif des modèles de langage, avec intervalles de Wilson. |
| `comparer_versions.py` | Comparaison appariée de deux versions sur les mêmes questions (test du signe). |
| `analyser_justification.py` | Mesure du contrôle par inférence textuelle : seuil choisi sur dev, appliqué une fois sur test, test de McNemar. |
| `verifier_application.py` | Vérifie une application en marche, locale ou distante, par son API, avec la seule bibliothèque standard de Python. |

## 7. Tests automatisés (`tests/`)

Les tests s'exécutent sans Ollama ni Internet : le modèle de langage est
simulé. Ils tournent sur GitHub à chaque envoi, et le Space n'est mis à jour
que s'ils réussissent.

| Fichier | Ce qu'il vérifie |
|---|---|
| `test_api.py` | Les routes de l'API et la tolérance aux fautes de frappe dans le nom du modèle. |
| `test_retrieval.py` | La recherche hybride, y compris des cas de non-régression. |
| `test_embedding_retrieval.py` | La recherche par embeddings, le repli sur la recherche hybride et l'exposition dans l'API. |
| `test_generation.py` | La génération typée, la vérification des extraits et le choix du statut. |
| `test_grounding.py` | Qu'une phrase qui ne vient pas des sources est bien masquée. |
| `test_definitions.py` | Que la définition est jointe quand une source y renvoie. |
| `test_claims.py` | Le jeu fictif, l'exclusion des dossiers en attente du taux d'acceptation, l'entraînement sur les seuls dossiers clos et la reproductibilité. |
| `test_decision.py` | L'aide à la décision : score expliqué, règles métier, clauses, orientation. |
| `test_project.py` | La cohérence du dépôt, du déploiement et de l'interface. |
| `conftest.py`, `__init__.py` | Préparation commune des tests. |
| `pytest.ini` (à la racine) | Configuration de pytest : dossier des tests et chemin d'import. |

## 8. Documentation (`docs/`, racine)

| Fichier | Contenu |
|---|---|
| `README.md` | Présentation, routes, organisation, tests, démarrage local, configuration, limites. Son en-tête configure aussi le Space. |
| `CHANGELOG.md` | Historique des versions, avec les mesures qui ont décidé chaque changement. |
| `docs/rapport/Rapport_Insurance_Claims_Assistance_v19.docx` | Rapport du projet (version 19). |
| `docs/rapport/Addendum_19.5.0_decisions_mesurees.pdf`, `docs/decisions_mesurees.md` | Les décisions prises depuis la version 19 et leurs mesures. |
| `docs/demonstration/demonstration_commentee.pdf` | Les cinq questions commentées (en local et en ligne), le score des sinistres et le lien avec les articles de référence. |
| `docs/guide_examinateur.md` | Vérification de l'application en dix minutes. |
| `docs/lancer_application.md`, `docs/Lancer_application_19.5.0.pdf` | Les six façons de faire tourner l'application, avec les commandes exactes. |
| `docs/fichiers_du_projet.md` | Ce document. |
| `docs/cadrage/` | Documents de cadrage du projet. |
| `docs/historique/` | Notes, changelogs et rapports des versions précédentes. |
