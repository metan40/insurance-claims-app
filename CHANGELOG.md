# Journal des versions

Les journaux détaillés des versions 4 à 18 sont conservés dans
`docs/historique/changelogs/`.

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
