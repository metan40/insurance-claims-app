# Décisions mesurées — Insurance Claims Assistance 19.5.0

Ce document résume les choix techniques de l'assistant documentaire et la
mesure qui a décidé chacun d'eux. Règle appliquée à chaque changement : il est
publié seulement si la mesure montre une amélioration **sans perte de bonnes
réponses** ; sinon il est documenté et non intégré.

## 1. Objectif et périmètre

- Aider un gestionnaire de sinistres à retrouver, dans 201 documents
  d'assurance habitation en français, ce qui répond à sa question, avec les
  extraits qui le prouvent.
- Exigence centrale : **aucune affirmation hors des sources**. Une réponse
  incomplète ou un « je ne sais pas » étayé vaut mieux qu'une conclusion
  inventée.
- L'application n'automatise aucune décision : elle prépare la lecture.

## 2. Protocole commun

- **Jeux de questions séparés** : un jeu de développement (réglages) et un jeu
  tenu à part (30 questions reformulées avec leur document de référence),
  utilisé une seule fois par décision.
- **Questions hors des documents** (30, réparties à l'avance en dev / test) :
  autres branches d'assurance, sujets habitation absents du corpus, questions
  hors assurance. Leur absence a été vérifiée par recherche dans le corpus.
- **Réponse conforme** : la réponse contient tous les faits obligatoires de la
  référence et cite le bon document.
- **Statistiques** : intervalle de Wilson à 95 % pour chaque proportion ; test
  de McNemar exact pour comparer deux versions sur les mêmes questions (paires
  discordantes seulement). Les passages répétés d'une même question ne sont
  pas indépendants : les intervalles sont donc légèrement optimistes.

## 3. Décisions

| # | Question posée | Mesure | Résultat | Décision |
|---|---|---|---|---|
| 1 | La recherche trouve-t-elle le bon document ? | Recall@3 sur 30 questions tenues à part | hybride TF-IDF + lexical 13/30 (43 %) ; bge-m3 26/30 (87 %, [70 % ; 95 %]) ; McNemar 14 contre 1, p = 0,001 | **bge-m3 retenu** (la fusion avec TF-IDF faisait moins bien : 70 %) |
| 2 | Quel modèle de langage avec bge-m3 ? | Réponses conformes, 30 questions × 3 passages | `llama3.2:1b` 29/90 (32 %) ; `qwen2.5:3b` 30/90 (33 %) ; `llama3.2:3b` 41/90 (46 %) | **`llama3.2:3b` retenu** |
| 2 bis | Bout en bout, 19.5.0 (bge-m3) fait-elle mieux que 19.4.0 (hybride), même modèle `llama3.2:3b` ? | Réponses conformes, mêmes 30 questions × 3 passages ; test du signe exact par question | 28/90 (31 %) → 41/90 (46 %) ; meilleure sur 8 questions, moins bonne sur 2, p = 0,11 ; refus corrects hors documents 7/56 → 9/56 | gain non significatif au seuil de 5 % ; deux régressions identifiées |
| 3 | Une conclusion peut-elle perdre sa condition ? | Cas réel « vélo volé dans la cave » | condition « formules 2 et 3, pas 1 » absente de la réponse | condition affichée sous la réponse, telle qu'écrite dans la source (19.4.0) |
| 4 | Le modèle conclut-il quand la réponse n'est pas dans les documents ? | Refus corrects, moitié test | 5/28 (18 %, [8 % ; 36 %]) | **limite documentée** |
| 5 | Un contrôle par inférence textuelle (mDeBERTa XNLI), prémisses = extraits cités, corrige-t-il la limite 4 ? | Seuil choisi sur dev (0,7), appliqué une fois sur test | refus 5/28 → 25/28 (p < 0,0001) ; réponses conformes 15/30 → 4/30 (11 perdues, p = 0,001) | **non intégré** |
| 6 | Même contrôle, prémisses = documents cités entiers (fenêtres de 3 phrases) ? | Seuil choisi sur dev (0,95), appliqué une fois sur test | refus 5/28 → 24/28 (86 %, [69 % ; 94 %]) ; réponses conformes 15/30 → 8/30 (7 perdues, p = 0,016) | **non intégré** |

Lecture des décisions 5 et 6 : le contrôle bloque presque toutes les
conclusions inventées, mais il ne distingue pas une reformulation correcte
d'une invention ; il ferait perdre une bonne réponse sur deux. Sur dev, même
le seuil le plus bas (0,05) perd déjà une bonne réponse. La bonne direction est
de vérifier, **avant** de rédiger, que le sujet de la question est traité par
les documents.

## 4. Questions de démonstration

Mesure : 5 passages par question, `llama3.2:3b` + bge-m3, application lancée
dans son conteneur et interrogée par son API (`evaluation/verifier_application.py`).

| Comportement attendu | Question | Résultat mesuré |
|---|---|---|
| Réponse complète | Que signifie louer un logement équipé de mobilier ? | 5/5 conformes, bon document cité |
| Réponse partielle avec restriction | La moisissure sur mes murs est-elle prise en charge ? | 5/5, restriction de la source affichée |
| Réponse partielle | Mon emplacement de stationnement doit-il être assuré ? | 4/5 |
| Absente des documents | Quel est le délai de carence de mon assurance décès ? | 0/5 : le modèle conclut (« 30 jours ») — illustre la limite 4 |
| « Je ne sais pas » étayé | Bougie laissée sur la table de jardin pendant les vacances | 2/5 en mode secours (extraits seuls) ; sinon conclusion sans rapport |

## 5. Comment lire une réponse

| Élément affiché | Signification |
|---|---|
| Statut « Réponse trouvée et justifiée » | le modèle conclut et tous ses extraits sont retrouvés mot pour mot dans les sources |
| Statut « Réponse partielle » | la source ne couvre qu'une partie de la question, ou impose une condition |
| Statut « Réponse absente des documents » | aucune source ne répond ; rien n'est affirmé |
| Statut « Réponse non vérifiable : extraits affichés » (mode secours) | la synthèse contenait une affirmation non appuyée : seuls les extraits vérifiés sont montrés |
| Étiquettes [S1], [S2]… | renvoient aux sources listées sous la réponse |
| « Condition indiquée par la source » | restriction recopiée de la source, à lire avant toute conclusion |
| Similarité sémantique | proximité entre la question et le document (bge-m3), pas une probabilité d'exactitude |

## 6. Reproduire

- Tests : `python -m pytest -v` (sans Ollama ni Internet).
- Application locale : voir « Démarrage local » dans le README (Docker, trois
  commandes), puis `python evaluation/verifier_application.py --url http://localhost:7860`.
- Mesures : `evaluation/mesurer_statuts.py`, `evaluation/analyser_justification.py`
  et les workflows GitHub « Évaluation de l'assistant par modèle »,
  « Mesure de la justification », « Vérification locale ».
