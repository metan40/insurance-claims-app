# Journal des versions

Les journaux détaillés des versions 4 à 18 sont conservés dans
`docs/historique/changelogs/`.

## 19.5.0 — recherche par embeddings multilingues (bge-m3)

### Choix du modèle et fonctionnement local (13/11/2026)
- Évaluation de bout en bout avec bge-m3 (30 questions tenues à part, 3
  passages) : réponses conformes à la référence 32 % avec `llama3.2:1b`,
  33 % avec `qwen2.5:3b`, 46 % avec `llama3.2:3b`. Le conteneur utilise
  désormais `llama3.2:3b` par défaut.
- Le dossier `/data` de l'image appartient à l'utilisateur de l'application :
  un volume Docker local conserve les modèles et l'index entre deux démarrages.
- Nom de modèle tolérant aux espaces et points finaux (incident réel :
  « llama3.2:3b. » saisi dans les variables du Space rendait `/ready` négatif).
- `evaluation/verifier_application.py` interroge une application en marche
  par son API HTTP (locale ou distante) ; `evaluation/questions_jury.csv`
  contient les questions candidates pour la démonstration (réponse complète,
  partielle, absente des documents).
- Workflow « Vérification locale » : construit l'image et la lance sur une
  machine vierge, puis pose ces questions, pour prouver que l'application
  fonctionne sans le serveur distant.

### Constat mesuré (12/11/2026)
- Évaluation de bout en bout (3 modèles de langage, 30 questions tenues à
  part) : la bonne fiche n'était parmi les 3 sources lues par le modèle que
  12 à 13 fois sur 30, quel que soit le modèle. La recherche limitait tout.
- Expérience `evaluation/experience_embeddings.py` (configuration choisie sur
  les 40 reformulations de développement, testée une seule fois sur les 30
  questions tenues à part) : Recall@3 de 13/30 (43 %) avec la recherche
  hybride à 26/30 (87 %, IC de Wilson 95 % [70 % ; 95 %]) avec bge-m3 seul ;
  McNemar exact apparié : 14 contre 1, p = 0,001. La fusion avec TF-IDF
  faisait moins bien que bge-m3 seul (70 %).

### Changements
- `src/embedding_retrieval.py` : classement par similarité cosinus des
  embeddings bge-m3 (titre + 1 000 premiers caractères), index mis en cache
  sur disque, chargement en arrière-plan.
- `src/app.py` : moteur choisi par `RETRIEVAL_ENGINE` (`embeddings` dans
  l'image Docker, `hybrid` par défaut ailleurs) ; tant que l'index n'est pas
  prêt, repli sur la recherche hybride, signalé dans la réponse
  (`retrieval_engine`) et dans `/ready` (`embedding_search_loaded`).
- Interface : affichage de la similarité sémantique bge-m3.
- `requirements.txt` : PyTorch CPU et sentence-transformers.
- Évaluation : `mesurer_statuts.py` attend l'index ; le workflow d'évaluation
  accepte le moteur en paramètre.
- Tests : `tests/test_embedding_retrieval.py` (encodeur factice).

## 19.4.0 — une conclusion n'est jamais affichée sans sa condition

### Constat (Space, 12/11/2026, version 19.3.0)
- Vélo volé dans la cave : « Si mon vélo a été volé, je suis couvert. [S2] »,
  alors que l'extrait vérifié de S2 précise « dans le cadre des formules n°2
  et 3 (voir conditions), mais pas n°1 ». Tous les mots venaient de la
  question ou de la source : le contrôle d'ancrage acceptait la phrase, mais la
  condition avait disparu.

### Changements
- `src/grounding.py` : `missing_conditions` repère les extraits vérifiés qui
  contiennent une restriction (« mais pas », « sauf », « ne sont pas »,
  « formule »…) ou des chiffres absents de la réponse affichée.
- `src/generation.py` : la condition est alors affichée sous la réponse, telle
  qu'elle est écrite dans la source (« Condition indiquée par la source : … »),
  et le statut passe à `partial`. Audit : `conditions_added_from_sources`.
- Tests : deux cas dans `tests/test_grounding.py` ; le test
  `test_partial_answer_at_top1_reads_three_sources` attend désormais la
  restriction de S2 sous la réponse (statut `partial`).

## 19.3.0 — la définition est jointe aux sources qui y renvoient

### Constat (Space, 12/11/2026, version 19.2.0)
- Vélo volé dans la cave : aucune phrase inventée n'est plus affichée, mais la
  réponse dépend de la définition des « objets à risque de vol » (un vélo en
  fait-il partie ?). Les sources lues y renvoyaient par « (voir définition) »
  sans que la fiche de définition soit parmi elles.

### Changements
- Nouveau module `src/definitions.py` : quand une source contient
  « <terme> (voir définition) » et qu'une fiche du corpus définit ce terme
  (« Que sont les… », « Qu'est-ce qu'un… »), son texte est joint à la fin de la
  source, sous la mention « Définition jointe automatiquement ». Le classement
  n'est pas modifié.
- `src/app.py` : jonction appliquée aux sources avant la génération.
- Tests : `tests/test_definitions.py` (index des fiches, renvois du corpus réel).

## 19.2.0 — aucune phrase non étayée n'est affichée

### Constat (Space, 12/11/2026)
- Question : vélo volé dans la cave. Réponse affichée : « Je suis couvert, car
  je transportais mon vélo dans un véhicule assuré. [S2] ». L'extrait cité de
  S2 était bien vérifié, mais « transportais » et « véhicule » venaient de S3
  (déménagement), une source que le modèle n'avait pas citée. La vérification
  des extraits prouvait que la citation existait, pas que la réponse en
  découlait.

### Changements
- Nouveau module `src/grounding.py` : contrôle d'ancrage de chaque phrase de
  la réponse. Un mot significatif doit venir de la question, d'une source dont
  un extrait est vérifié, ou d'un petit vocabulaire de rédaction (comparaison
  sur les 5 premières lettres, sans accents). Une phrase qui apporte au moins
  deux mots nouveaux est jugée non étayée.
- `src/generation.py` : une réponse dont aucune phrase n'est étayée est traitée
  comme invalide (étape suivante de la boucle) ; à la dernière étape,
  l'application affiche les extraits vérifiés, jamais la synthèse (statut
  `fallback`). Une réponse en partie étayée passe en `partial` : seules ses
  phrases étayées sont affichées, avec le nombre de phrases masquées. Les
  phrases masquées restent dans l'audit (`unsupported_sentences`,
  `model_unverified_text`).
- Tests : `tests/test_grounding.py` (cas réel du vélo, réponse fidèle conservée,
  masquage partiel).

### Limite
- Contrôle lexical : il détecte une information apportée d'ailleurs, pas une
  négation inversée construite avec les mêmes mots.

## 19.1.0 — l'assistant cite avant de conclure

### Constat
- L'assistant affichait souvent « réponse absente des documents » alors que la
  bonne source était lue. Le modèle (llama3.2:1b) écrit le JSON dans l'ordre
  du schéma : `answer_found` venait en premier, si bien qu'il tranchait
  « trouvé / non trouvé » avant d'avoir recopié le moindre extrait. Les champs
  facultatifs (`evidence`, `caveats`) étaient en outre relégués en fin de
  sortie par la grammaire JSON d'Ollama.

### Changements (`src/generation.py`)
- Nouvel ordre du schéma `TypedAnswer` : `evidence`, `answer`, `caveats`,
  `answer_found`, `complete_answer_found`, `confidence` ; tous les champs
  sont requis pour que cet ordre soit respecté.
- Consigne réécrite dans le même ordre. Une réponse qui dépend d'une
  condition (formule, type de bien, exclusion) est une réponse partielle
  dont les conditions sont données, et non une réponse absente.
- Consigne de citation réécrite : mesuré sur un poste local, llama3.2:1b
  renvoyait littéralement « MOT POUR MOT » comme extrait, recopiant la
  consigne au lieu d'une phrase du document (0 extrait vérifié sur 2). La
  consigne dit désormais « copie chaque phrase telle qu'elle est écrite dans
  le document » et interdit de copier une phrase des consignes ; un extrait
  repris de la consigne est signalé (`copied_from_instructions`) et compté
  dans le journal de diagnostic.
- Sorties coupées : la même mesure montre 5 appels sur 10 interrompus en plein
  texte par la limite de 280 jetons, donc un JSON illisible. Le schéma transmis
  à Ollama borne désormais la longueur des champs (extrait 240 caractères,
  réponse 400, réserves 200) et `OLLAMA_NUM_PREDICT` passe à 400 par défaut.
- Boucle top-1 / top-3 : une réponse partielle vérifiée obtenue à l'étape 1
  est conservée si l'étape 2 renvoie une sortie illisible ou « non trouvé »
  (3 questions sur 5 perdaient ainsi une réponse justifiée).
- Modèle contradictoire (« non trouvé » avec des extraits vérifiés) : sa
  synthèse n'est plus affichée, seulement les extraits vérifiés ; elle restait
  parfois fausse (« Non, je ne suis pas couvert » alors que la source dit oui).
- Réserves (`caveats`) rédigées par le modèle : non vérifiées, elles ne sont
  plus affichées ni livrées comme information sûre ; elles restent dans
  l'audit (`model_unverified_caveats`). Mesure du 6/11 : « les dommages
  causés par les enfants ne sont pas couverts », contraire à la source.
- Résultat sur les 5 questions de démonstration (poste local, llama3.2:1b) :
  avant, 0 réponse justifiée, 3 « absentes », 2 replis, 4 JSON illisibles sur
  10 appels ; après, 5 réponses appuyées par des extraits vérifiés, 0 JSON
  illisible, 28 s en moyenne au lieu de 19 s.
- La vérification des extraits et la décision du statut par l'application
  sont inchangées.

### Mesure
- Nouveau script `evaluation/mesurer_statuts.py` : statut de chaque question
  et, pour chaque appel, JSON valide, `answer_found`, extraits cités et
  vérifiés ; indique si la bonne source faisait partie des documents lus.
  À lancer avant et après le changement, sur les mêmes questions.

## 19.0.0 — nettoyage, tests et documentation d'examen

### Organisation du dépôt
- Suppression des copies en double à la racine (scripts, données, rapports,
  ancien `app.py` 7.5.0 divergent, `retrieval.py` vide, fichiers « (2) », « (3) »).
- Rangement par rôle : `src/` (application), `evaluation/`, `experiments/`
  (archives), `docs/` (guide examinateur, cadrage, historique).
- Le fichier nommé `download` contenait les règles d'exclusion Docker ; il est
  renommé `.dockerignore` et complété.
- Version unique dans `VERSION` (auparavant 18.0 dans VERSION.txt, 7.6.0 dans
  le code et 7.5.0 dans les tests).

### Alignement sur les recommandations du Groupe Foyer
- **Analyse de la question** (`src/question_parsing.py`) : dictionnaire de
  synonymes métier ; la question devient une liste de concepts. Recall@5 sur
  30 reformulations tenues à part : 36,7 % → 46,7 % ; sur le jeu de
  développement (40) : 40,0 % → 67,5 % ; titres : inchangé (89,6 %).
- **Réponse typée** (`src/generation.py`) : sortie JSON contrainte par schéma
  (`answer_found`, `complete_answer_found`, `answer`, `evidence`,
  `confidence`, `caveats`). L'application vérifie chaque extrait dans la
  source citée et décide du statut : `answered`, `partial`, `not_found`,
  `fallback`. Une réponse « absente des documents » est un résultat valide.
- **Boucle top-1 puis top-3** : la meilleure source seule (1 200 caractères),
  puis les trois meilleures (900 caractères) seulement si la réponse est
  incomplète ou non vérifiée. Texte lu par le modèle : − 64 % quand la
  première source suffit, − 50 % dans le pire cas (mesure de référence sur le
  Space en 7.6 : 228 s de génération). `num_predict` 350, `num_ctx` 2048.

- **Aide à la décision sur un dossier** (`src/claim_decision.py`,
  `GET /api/claims/{id}/decision-support`) : score expliqué par les
  contributions de la régression logistique, règles métier explicites
  (dossier incomplet, garantie à vérifier, montant au-delà du 90e centile,
  historique chargé, Pending de plus de 30 ou 90 jours), clauses du contrat
  par type de sinistre (table de correspondance validable par un expert),
  signalement des trous du corpus (incendie, bris de glace) et orientation
  pour le gestionnaire. Chaque clause ouvre l'assistant RAG.

- **Correction après essai réel** : à score lexical égal, le départage
  retient le titre le plus centré sur la question (au lieu du plus long) ;
  la fiche de définition de la responsabilité civile revient dans le top 3.
  Recall@5 sur le jeu tenu à part : 53,3 %. Temps mesuré sur le Space :
  225 s (deux appels) ; sortie réduite à 280 jetons et deux extraits.

- **Choix du modèle par la mesure** (même question, sur le Space) :
  `llama3.2:3b` 260 s, réponse partielle cohérente ; `llama3.2:1b` 115 s,
  réponse correcte mais indicateurs contradictoires (« non trouvé » avec des
  extraits vérifiés). Modèle par défaut : `llama3.2:1b`. L'application
  tranche désormais : un « non trouvé » accompagné d'extraits vérifiés devient
  « partielle » et l'incohérence est affichée. Entités HTML décodées.

### Application
- Recherche extraite dans `src/retrieval.py` (`HybridRetriever`), avec un
  classement identique à la 7.6.0 vérifié sur 244 questions.
- Libellés corrigés : le pipeline n'utilise plus d'embeddings ni de
  CrossEncoder ; les champs `score`/`reranker_score`/`embedding_similarity`
  sont remplacés par `tfidf_cosine_similarity`.
- Rapidité : `num_predict` (400) borne la longueur des réponses,
  `keep_alive` (30 min) évite de recharger le modèle, et le modèle est
  préchargé au démarrage du conteneur.
- `/ready` compare exactement le modèle installé (`llama3.2:1b` n'est plus
  confondu avec `llama3.2:3b`).
- L'interface affiche le message d'erreur du serveur (ex. Ollama indisponible).
- Image Docker allégée : `sentence-transformers`/PyTorch retirés des
  dépendances d'exécution ; seuls `src/`, `static/` et `Data/` sont copiés.

### Qualité
- Suite de tests réorganisée en 5 fichiers (recherche, génération, API,
  sinistres, cohérence du projet), dont des tests HTTP de bout en bout.
- Évaluation reproductible `evaluation/evaluate_pipeline.py` sur les 201
  titres et sur 40 questions reformulées.
- GitHub Actions : tests à chaque push ; synchronisation Hugging Face depuis
  `main` uniquement si les tests réussissent.
- Guide de vérification pas à pas pour l'examinateur.
