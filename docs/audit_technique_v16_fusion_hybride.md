# Audit technique V16 — fusion hybride du classement documentaire

## 1. Synthèse

J'ai examiné la fusion hybride utilisée par le moteur RAG après l'audit documentaire de la V15. Cette analyse a mis en évidence un défaut précis : le rang lexical contribuait au score final même lorsque le document ne présentait aucune correspondance lexicale avec la question. J'ai corrigé ce comportement sans remplacer les modèles existants et sans supprimer les fonctionnalités de l'application.

La V16 constitue un correctif ciblé de la logique de classement. Les tests automatisés vérifient la nouvelle règle, mais ils ne permettent pas de conclure que la pertinence globale des résultats a augmenté. Cette conclusion nécessite une évaluation contrôlée sur les questions de référence.

## 2. Architecture examinée

L'application encode la question à l'aide du modèle `paraphrase-multilingual-MiniLM-L12-v2`. Elle calcule la similarité cosinus entre l'embedding de la question et les embeddings des documents. Elle complète ensuite le signal sémantique par un score lexical qui valorise les termes présents dans les titres et les contenus. Le CrossEncoder `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reclasse les candidats retenus. L'application transmet enfin jusqu'à cinq documents au modèle Ollama pour produire une réponse accompagnée de références.

Chaque composant répond à un objectif différent. Les embeddings favorisent la proximité sémantique, le score lexical favorise la correspondance explicite des termes et le CrossEncoder estime la pertinence d'une paire question-document. Les scores de ces composants ne sont pas interchangeables et ne doivent pas être interprétés comme des probabilités sans calibration spécifique.

## 3. Défaut de classement identifié

La fonction de classement attribuait un rang lexical à tous les documents. Ce comportement est normal lorsqu'une liste est triée, mais il devient problématique lorsque la fusion traite chaque rang comme un signal utile. Un document sans aucun terme commun avec la question possède malgré tout un rang lexical, même si ce rang résulte seulement de l'ordre de départage entre documents ayant tous un score nul.

Dans cette situation, la contribution lexicale n'exprime pas une correspondance documentaire. Elle introduit un terme numérique sans signification pertinente pour la requête. Ce terme peut modifier l'ordre de deux documents et rendre l'explication du classement moins fidèle aux signaux réellement observés.

## 4. Formalisation mathématique

La fusion réciproque de rangs utilise généralement une contribution de la forme :

\[
\operatorname{RRF}(d)=\sum_{j=1}^{m}\frac{1}{k+r_j(d)},
\]

où \(r_j(d)\) désigne le rang du document dans la liste \(j\), \(k\) désigne une constante de stabilisation et \(m\) désigne le nombre de listes fusionnées.

Pour le pipeline concerné, j'ai rendu la contribution lexicale conditionnelle au score lexical :

\[
H(d)=\frac{1}{k+r_p(d)}+\mathbb{1}[s_l(d)>0]\frac{1}{k+r_l(d)}.
\]

Dans cette expression, \(r_p(d)\) désigne le rang principal, \(r_l(d)\) désigne le rang lexical et \(s_l(d)\) désigne le score lexical. L'indicatrice vaut un lorsque le score lexical est positif et zéro lorsqu'il est nul. La première contribution reste active pour chaque candidat classé par le signal principal. La seconde contribution n'est ajoutée que lorsqu'un recouvrement lexical a effectivement été détecté.

Cette règle ne prouve pas que les mots partagés expriment le même sens. Elle garantit seulement qu'une absence de correspondance lexicale ne reçoit pas de bonus lexical artificiel. Les synonymes, les paraphrases, la morphologie et la négation restent des limites du score lexical actuel.

## 5. Modifications intégrées

J'ai ajouté la fonction `_rank_fusion_score` pour centraliser la formule et éviter que plusieurs parties du code appliquent des variantes différentes. J'ai utilisé cette fonction lors de la présélection hybride et lors du classement final après le CrossEncoder. J'ai également aligné le champ `hybrid_rank_score` de la réponse API sur la formule appliquée par le classement.

J'ai conservé le modèle d'embeddings, le CrossEncoder, le corpus documentaire, le nombre maximal de documents retournés, la génération Ollama et l'interface. Cette portée limitée permet d'isoler le changement et facilite l'analyse de ses effets.

## 6. Tests réalisés

J'ai ajouté deux tests unitaires. Le premier vérifie qu'un score lexical nul ne contribue pas à la fusion, même si un rang lexical existe. Le second vérifie qu'un score lexical positif ajoute la contribution attendue. La suite complète affiche 42 tests réussis. La compilation des modules Python réussit également.

Les tests utilisent des doubles pour les modèles et n'exécutent pas le pipeline complet avec les poids réels. Ils vérifient la logique logicielle de la formule, mais ils ne mesurent pas les résultats de recherche sur le corpus. Je ne déduis donc aucun gain de Recall@5 ou de MRR à partir de ces tests.

## 7. Contrôles quantitatifs à effectuer

Pour établir un résultat empirique, je dois rejouer les mêmes 201 questions de référence avant et après le correctif. Le corpus, les modèles, les paramètres et les règles d'annotation doivent rester constants. Je dois calculer Recall@1, Recall@3, Recall@5, Recall@20 et MRR pour chaque version. Je dois ensuite analyser les cas améliorés et les cas dégradés, car une moyenne agrégée peut masquer des erreurs importantes sur certaines catégories de questions.

Je dois également mesurer la durée de l'embedding, de la récupération, du reranking et de la génération. Une seule requête ne suffit pas pour estimer une latence représentative. Je dois effectuer plusieurs répétitions dans un environnement documenté et comparer les médianes ainsi que la dispersion des durées.

Enfin, je dois vérifier manuellement la fidélité des réponses. La présence d'un identifiant tel que `[S1]` confirme seulement qu'une référence a été émise et reconnue. Elle ne démontre pas que le contenu de S1 soutient chaque affirmation de la réponse.

## 8. Suivi de maintenance

Le fichier `src/app.py` est passé de 726 à 754 lignes, soit 28 lignes supplémentaires. Le fichier `tests/test_app_validation.py` est passé de 497 à 521 lignes, soit 24 lignes supplémentaires. Le rapport d'évolution conserve ses sections antérieures et inclut une analyse de la V16. Le correctif respecte ainsi la règle de maintenance qui privilégie l'enrichissement sans réduction de la taille des fichiers concernés.

## 9. Conclusion et statut

La V16 corrige une incohérence de la fusion hybride en supprimant la contribution lexicale lorsque le score lexical est nul. Les tests automatisés valident le comportement de la fonction de fusion dans les cas couverts. La qualité réelle du classement reste à mesurer sur le jeu de questions annotées, et la génération doit encore être vérifiée avec les modèles réels.

Je considère donc que le correctif est testé au niveau unitaire, mais que l'amélioration de la pertinence globale n'est pas encore démontrée. Cette distinction entre correction de code, résultat mesuré et validation de bout en bout rend le rapport plus transparent et permet de présenter à Khezan un état d'avancement défendable.
