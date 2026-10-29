# Audit technique de la V14 et plan d'amélioration de la V15

## 1. Objet et périmètre de l'audit

J'ai examiné l'archive `insurance_claims_app_v14_ready_for_keyzan.zip` afin d'établir un point de départ vérifiable avant toute modification du moteur de recherche documentaire. L'audit couvre la structure du projet, le code du pipeline RAG, les tests automatisés, les fichiers d'évaluation de la récupération et la documentation de livraison.

L'archive contient 62 fichiers. Le contrôle d'intégrité ZIP ne signale aucune erreur. Le fichier `src/app.py` contient 726 lignes, le fichier `tests/test_app_validation.py` contient 497 lignes et le rapport historique `docs/rapport_evolution_versions.md` contient 290 lignes. Ces valeurs constituent les références de départ pour suivre les changements. La règle de maintenance retenue consiste à enrichir les fichiers sans réduire leur nombre de lignes et sans supprimer les fonctionnalités déjà présentes.

J'ai exécuté la suite automatisée depuis la racine du projet. Le résultat est de **40 tests réussis en 1,47 seconde**. Ce résultat confirme que les comportements couverts par ces tests restent cohérents dans cet environnement. Il ne démontre toutefois pas, à lui seul, la pertinence réelle des documents renvoyés par le modèle d'embeddings et le CrossEncoder, ni la fidélité factuelle des réponses générées par un serveur Ollama réel.

## 2. Architecture observée

Le pipeline implémenté suit plusieurs étapes. L'application encode d'abord la question avec `paraphrase-multilingual-MiniLM-L12-v2`. Elle compare ensuite cet embedding aux embeddings pré-calculés du corpus au moyen de la similarité cosinus. Une sélection hybride combine les rangs sémantiques et lexicaux. Le CrossEncoder `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reclasse les candidats. Une fusion de rangs s'applique ensuite avant de transmettre jusqu'à cinq documents au modèle Ollama pour la génération.

Cette architecture combine trois signaux distincts : la proximité sémantique, la correspondance lexicale et la pertinence estimée par le CrossEncoder. Cette séparation est pertinente parce que ces signaux n'ont ni la même définition ni la même échelle. En particulier, un score de CrossEncoder peut être négatif et ne doit pas être interprété comme une probabilité de pertinence.

La fonction `audit_citations` vérifie si la réponse cite des identifiants de source existants. Son contrat précise explicitement qu'elle ne vérifie pas si les passages cités justifient réellement les affirmations. Cette distinction est essentielle : une citation formellement valide n'est pas nécessairement une preuve suffisante.

## 3. Résultats quantitatifs disponibles

Le fichier `Data/retrieval_optimization_detailed.csv` contient 201 questions de référence avec le rang du document attendu avant et après une optimisation historique.

| Indicateur historique | Résultat calculé |
|---|---:|
| Nombre de questions évaluées | 201 |
| Document attendu dans le Top 5 avec les embeddings | 67,16 % |
| Document attendu dans le Top 5 après l'optimisation enregistrée | 76,12 % |
| Variation absolue du Top 5 | +8,96 points de pourcentage |
| Document attendu présent dans le Top 20 après l'optimisation | 83,08 % |
| Document attendu absent du Top 20 après l'optimisation | 34 questions, soit 16,92 % |
| Questions classées comme améliorées dans le fichier | 68 |
| Questions classées comme dégradées dans le fichier | 23 |
| Questions classées au même rang dans le fichier | 76 |

Le taux Top 5 est calculé comme le nombre de questions dont le document attendu est présent parmi les cinq premiers résultats, divisé par 201. La différence de 8,96 points de pourcentage représente la soustraction du taux initial au taux optimisé ; elle ne représente pas une amélioration relative de 8,96 %.

Ces résultats sont encourageants, mais ils doivent être interprétés avec prudence. Ils proviennent d'un fichier d'évaluation déjà enregistré et non d'une nouvelle exécution indépendante du pipeline complet de la V14. Avant d'utiliser ces chiffres comme preuve définitive d'une amélioration, il faut vérifier que les paramètres, le corpus, les modèles et la définition du classement correspondent exactement à la version évaluée. Il faut également inspecter les questions qui se dégradent et les questions pour lesquelles le document de référence reste absent du Top 20.

## 4. Analyse mathématique des signaux de classement

### 4.1 Similarité cosinus

Pour une question représentée par le vecteur \(q\) et un document représenté par le vecteur \(d\), la similarité cosinus est définie par :

\[
\operatorname{cos}(q,d)=\frac{q^\top d}{\|q\|_2\,\|d\|_2}.
\]

Lorsque les deux vecteurs sont normalisés en norme euclidienne, cette expression se réduit au produit scalaire \(q^\top d\). Le score mesure un alignement entre représentations vectorielles. Il ne garantit pas que le document contienne la règle contractuelle recherchée. Deux textes peuvent être sémantiquement proches tout en répondant à des questions métier différentes.

### 4.2 Signal lexical

La fonction `_lexical_relevance_score` normalise les accents et la ponctuation, supprime certains mots fréquents, puis calcule la couverture des termes de la question dans le titre et le contenu. Elle accorde davantage de poids aux correspondances dans le titre et ajoute un bonus pour certaines paires de termes consécutifs.

Ce signal peut aider à retrouver des expressions contractuelles précises comme « franchise absolue ». Cependant, il reste une heuristique lexicale : il ne modélise pas complètement les synonymes, les variantes morphologiques, les négations ni les relations juridiques entre les termes. Un document qui répète les mots de la question peut obtenir un score élevé sans fournir la réponse attendue. Il convient donc d'évaluer les faux positifs, et pas seulement les documents retrouvés.

### 4.3 Reciprocal Rank Fusion

La fusion de rangs utilise la forme générale suivante :

\[
\operatorname{RRF}(d)=\sum_{j=1}^{m}\frac{1}{k+r_j(d)},
\]

où \(r_j(d)\) est le rang du document \(d\) dans la liste \(j\), \(k\) est une constante de stabilisation et \(m\) représente le nombre de listes fusionnées. L'intérêt de cette méthode est de combiner des classements sans additionner directement des scores qui ont des échelles différentes.

La V14 utilise une constante de fusion de 60 et combine les rangs du retriever et le rang lexical lors de la présélection, puis le rang du CrossEncoder et le rang lexical dans le classement final. Cette conception rend la fusion explicable, mais elle impose un contrôle empirique : le signal lexical peut améliorer la présence des termes métier tout en promouvant des documents lexicalement proches mais insuffisants sur le plan sémantique. La valeur de la constante et la contribution des listes doivent être comparées sur un jeu de validation indépendant.

### 4.4 Score du CrossEncoder

Le CrossEncoder évalue conjointement la question et chaque document candidat. Il peut donc estimer leur pertinence relative plus finement qu'une simple comparaison de deux embeddings indépendants, mais son score brut n'est pas une probabilité calibrée. Il faut comparer l'ordre qu'il produit avec les jugements de pertinence humains et mesurer le coût temporel de l'inférence.

## 5. Diagnostic des risques

Le premier risque est la récupération d'un document proche du thème général, mais qui ne répond pas au point précis demandé. Le deuxième risque est la survalorisation d'une correspondance lexicale partielle. Le troisième risque est la confusion entre validité syntaxique d'une citation et soutien factuel de l'affirmation. Le quatrième risque est de considérer une augmentation du Top 5 comme suffisante sans analyser les cas dégradés et les questions sans document pertinent dans le Top 20.

Le contrôle de citation existant permet de repérer des identifiants absents ou inconnus. Il ne vérifie pas l'implication logique entre chaque phrase de la réponse et l'extrait source. Une amélioration sérieuse doit donc distinguer au moins trois dimensions : la pertinence du contexte récupéré, la fidélité factuelle de la réponse et la validité des références.

## 6. Plan d'évaluation recommandé pour la suite

Je recommande de constituer un jeu de questions annotées qui couvre les notions contractuelles exactes, les paraphrases, les questions ambiguës, les questions hors corpus et les termes fréquemment confondus. Chaque question doit être associée à un ou plusieurs documents de référence et, si possible, aux passages précis qui justifient la réponse.

Le protocole doit calculer Recall@k, Precision@k, MRR et nDCG pour la récupération. Il doit mesurer séparément le taux de citations valides, le soutien factuel des affirmations et le taux de réponses injustifiées pour la génération. Les temps d'embedding, de récupération, de reranking, de génération et le temps total doivent être relevés séparément. Les comparaisons doivent utiliser les mêmes questions, le même corpus et les mêmes conditions de calcul.

Une amélioration ne doit être acceptée que si les résultats mesurés montrent un bénéfice sur l'indicateur visé sans dégradation inacceptable d'autres dimensions. Les seuils d'acceptation doivent être définis avant la comparaison, et les cas d'échec doivent être conservés pour éviter de masquer les régressions.

## 7. Contrôles effectués et limites de l'audit

Le ZIP est intègre et les 40 tests automatisés réussissent. Le code expose également les routes de santé, de disponibilité et de version. Ces vérifications sont utiles pour la recette opérationnelle, mais elles ne remplacent pas un test de bout en bout avec les modèles réels et le service Ollama disponible.

Le fichier `src/retrieval.py` est actuellement presque vide. Le pipeline de recherche est implémenté dans `src/app.py`. Cette situation ne bloque pas l'exécution actuelle, mais elle peut compliquer la maintenance et les tests unitaires indépendants du démarrage complet de l'application. Toute refactorisation éventuelle doit être additive et accompagnée de tests, sans supprimer les comportements existants.

Le nom de l'archive indique V14, tandis que `VERSION.txt` décrit la version applicative 7.5.0. Il faut distinguer explicitement la version de livraison du paquet et la version interne de l'application afin d'éviter toute ambiguïté pendant la recette.

## 8. Conclusion destinée à la revue technique

Mon audit confirme que la V14 dispose déjà d'une architecture RAG hybride, d'un reranker, d'un contrôle formel des citations, d'une instrumentation des durées et d'une suite de tests automatisés. Les résultats historiques suggèrent un gain de récupération dans le Top 5, mais ils montrent aussi que des documents attendus restent hors du Top 20 et que certaines questions subissent une dégradation.

Je considère donc que la prochaine étape doit prioriser une évaluation reproductible du retrieval et du reranking, puis un contrôle plus fort du soutien factuel des réponses. Cette démarche permettra de justifier les évolutions par des mesures, de documenter les limites résiduelles et de présenter à Keyzan une analyse honnête, explicable et techniquement défendable.
