# Évaluation mesurée de la récupération — V16

## 1. Objet de cette mesure

J'ai exécuté une évaluation lexicale indépendante sur le corpus `Data/insurance-fr.csv`. Le fichier contient 201 lignes et deux colonnes, `title` et `content`. Pour chaque ligne, le titre est utilisé comme question de référence, le contenu de chaque ligne est indexé avec TF-IDF et le document de la même ligne est considéré comme document attendu. Cette définition est pratique pour obtenir une première mesure reproductible, mais elle ne remplace pas un jeu de questions annotées par des experts métier.

## 2. Méthode

J'ai construit une matrice TF-IDF avec `TfidfVectorizer(lowercase=True)`. Pour chacune des 201 questions, j'ai calculé la similarité cosinus entre le vecteur de la question et les vecteurs des 201 contenus. J'ai ensuite classé les documents par similarité décroissante et calculé le rang du document attendu. Le protocole suit la logique du script `src/evaluate_retrieval.py` déjà présent dans le projet.

Le rappel à k, noté Recall@k, est ici la proportion de questions pour lesquelles le document attendu apparaît parmi les k premiers résultats. Le MRR (Mean Reciprocal Rank) est la moyenne de l'inverse du rang du document attendu :

\[
\operatorname{MRR}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i},
\]

où N vaut 201 et r_i désigne le rang du document attendu pour la question i.

## 3. Résultats observés

| Indicateur | Résultat mesuré |
|---|---:|
| Nombre de questions | 201 |
| Recall@1 | 29,85 % (60/201) |
| Recall@3 | 52,24 % (105/201) |
| Recall@5 | 62,19 % (125/201) |
| Recall@20 | 77,11 % (155/201) |
| MRR | 0,4352 |

Le fichier `Data/tfidf_reference_evaluation_v16.csv` conserve un résultat par question, le rang du document attendu, les indicateurs de présence dans les Top-k, le premier document retourné et sa similarité cosinus. Il permet d'inspecter les échecs question par question sans se limiter aux moyennes agrégées.

## 4. Interprétation et limites

Cette mesure établit une référence TF-IDF dans l'environnement d'exécution utilisé. Elle ne mesure pas le classement hybride de production de la V16, car elle n'exécute ni le modèle d'embeddings, ni le CrossEncoder, ni la fusion hybride de l'application. Elle n'évalue pas non plus la qualité des réponses générées par Ollama, la fidélité des citations ou la justesse métier des documents considérés comme pertinents.

Les titres sont utilisés tels quels comme questions et le document de la même ligne est considéré comme l'unique référence attendue. Or, d'autres documents peuvent être pertinents pour une même question. En conséquence, les chiffres doivent être interprétés comme des mesures de récupération d'une référence définie par la construction du corpus, et non comme une mesure exhaustive de pertinence sémantique.

Les fichiers historiques du projet rapportent, sur les mêmes 201 entrées, 67,16 % de présence du document attendu dans le Top 5 pour la colonne `embedding_top5` et 76,12 % pour la colonne `optimized_top5`. Ces valeurs sont des résultats historiques consignés dans `Data/retrieval_optimization_detailed.csv`; elles ne sont pas une nouvelle exécution du pipeline V16. Elles ne permettent donc pas de conclure que le correctif V16 améliore ou dégrade le classement.

Dans l'environnement utilisé pour cette mesure, le paquet `sentence_transformers` n'est pas installé. Je n'ai donc pas pu rejouer de manière contrôlée le pipeline avec les poids réels d'embeddings et du CrossEncoder. Je ne présente pas les résultats TF-IDF comme un substitut à cette validation. La comparaison avant/après V16 reste à réaliser dans un environnement disposant des dépendances et modèles requis, avec les mêmes questions, le même corpus et les mêmes paramètres.

## 5. Prochaine étape de validation

La prochaine expérience doit exécuter la version antérieure et la V16 dans un environnement identique. Elle doit enregistrer, pour chaque question, les rangs avant et après, les documents ajoutés ou retirés du Top-k, Recall@1/3/5/20, MRR et les temps d'exécution. Une revue qualitative doit ensuite analyser les cas améliorés et dégradés, notamment les questions d'assurance pour lesquelles une confusion entre franchise, exclusions, plafonds ou conditions de couverture pourrait produire une réponse trompeuse.

## 6. Conclusion

La référence TF-IDF a été exécutée et ses résultats par question ont été sauvegardés. Cette mesure fournit une base de diagnostic reproductible, mais elle ne valide pas encore la qualité du classement hybride V16. Le correctif de fusion reste validé au niveau unitaire par les tests automatisés; son effet empirique sur le RAG de production doit être établi séparément.
