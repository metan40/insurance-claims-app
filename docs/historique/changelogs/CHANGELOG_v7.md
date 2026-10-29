# Version 7 — rendre le pipeline RAG plus observable

## Constat

Les versions précédentes vérifiaient la validation des questions, les erreurs de génération et les références formelles aux sources. Cependant, elles ne renvoyaient pas de mesure de durée par étape. Pendant la campagne de tests, il serait donc difficile de distinguer le temps pris par l'embedding, la recherche, le reranking et la génération.

## Hypothèse et décision

J'ai ajouté des mesures de durée avec `time.perf_counter()`. Cette fonction sert à mesurer les durées écoulées au cours d'une requête. Les valeurs sont renvoyées dans `pipeline_metrics` et affichées dans l'interface. Je n'ai pas changé les modèles, le nombre maximal de candidats ni le nombre de documents transmis au générateur.

## Changements

- Ajout des durées `embedding_ms`, `retrieval_ms`, `reranking_ms`, `generation_ms` et `total_ms` dans la réponse JSON.
- Ajout des compteurs `document_count`, `candidate_count` et `returned_count` pour donner du contexte aux mesures.
- Affichage des durées dans l'interface, en rappelant qu'elles varient selon la machine et la charge.
- Ajout de tests automatisés pour la structure des métriques, leur affichage et la résolution des chemins depuis un autre répertoire de travail.
- Mise à jour du rapport d'évolution et du plan de validation.

## Résultats des tests automatisés

La suite compte 19 tests et les 19 ont réussi dans l'environnement de vérification. La compilation syntaxique des fichiers Python a également réussi. Les tests utilisent des doubles pour les modèles et Ollama ; les durées observées dans ces tests ne doivent pas être utilisées comme mesures de performance de production.

## Limites

Les mesures ne couvrent que les étapes instrumentées dans la fonction de recherche et ne constituent pas une analyse de performance exhaustive. Le temps de chargement initial des modèles n'est pas inclus dans `total_ms`. La validation avec les vrais modèles et un serveur Ollama actif reste à effectuer. Les tests de bout en bout devront relever les durées réelles sur plusieurs questions et dans un environnement défini.


## Correctif V7.1 — révision des réponses sans citations valides

### Constat observé pendant les essais réels

Lors d'une question sur la garantie optionnelle de revente, la réponse reprenait le contenu d'une source pertinente, mais aucune référence `[S<number>]` n'était fournie. Le contrôle existant détectait correctement l'absence de citations, sans toutefois empêcher l'affichage de cette réponse non référencée.

### Modification

- Après une première réponse sans citation reconnue ou contenant des identifiants inconnus, l'application demande une seule révision au modèle en lui rappelant les identifiants disponibles et l'obligation de citer chaque affirmation factuelle.
- Si la seconde réponse ne passe toujours pas le contrôle formel, l'application remplace le contenu généré par un message prudent indiquant qu'elle ne peut pas fournir une réponse suffisamment traçable.
- Les erreurs de connexion et les réponses Ollama mal formées conservent le traitement HTTP 502 existant.
- Le pipeline de recherche, les modèles, les 20 candidats initiaux et les cinq documents transmis restent inchangés.

### Vérifications automatisées

Dans la V7.1, la suite compte 24 tests, tous réussis lors de la vérification de l’archive. Les tests couvrent la révision après absence de citations et le repli prudent si la révision échoue. Ces tests utilisent un faux client Ollama ; ils vérifient la logique du code mais ne garantissent pas que le modèle réel respectera toujours la consigne.

### Limites

La vérification reste formelle : une citation valide ne prouve pas que chaque affirmation est soutenue par la source. La tentative de révision peut ajouter un deuxième appel à Ollama et donc augmenter la latence pour les réponses initialement mal référencées. Le comportement doit être revalidé dans l'interface avec le vrai modèle `llama3.2:3b`.


## Correctif de diagnostic — V7.1.0

- Ajout d’un endpoint `GET /version` qui renvoie la version de l’application et le nom du modèle Ollama configuré, afin de vérifier que l’instance lancée correspond bien à l’archive testée.
- Ajout du champ `app_version` aux réponses de recherche.
- Ce correctif de diagnostic ne change pas la logique de recherche, de reranking ou de citation.
- Le test réel précédent a encore renvoyé une réponse sans citation alors que le code de l’archive prévoit une seconde tentative et une réponse de repli. Il faut donc vérifier l’instance réellement lancée avant de conclure sur le comportement du correctif.


## Correctif V7.2.0 — contexte complet lors de la révision

### Constat

Le code de V7.1 demandait au second appel Ollama de s'appuyer sur « le prompt précédent ». Or, les appels successifs à `/api/generate` sont indépendants : le second appel ne reçoit pas automatiquement la question ni les documents du premier. Il ne disposait que de la consigne de révision, des identifiants autorisés et de la réponse à corriger.

### Modification

La seconde requête transmet explicitement la question originale, les documents récupérés avec leurs identifiants, les règles de citation et la réponse précédente. La recherche, le reranking, le modèle et le mécanisme de repli restent inchangés. La version exposée passe à `7.2.0`.

### Vérification

Le test automatisé de la nouvelle tentative vérifie maintenant que la seconde requête contient la question, le contenu source et l'identifiant exact. Vérification effectuée sur cette archive : `24 passed` et compilation Python réussie. Le test valide la construction du prompt, pas le comportement garanti du modèle réel.

## Essai réel V7.2.0 — résultat et suite à donner

- L'endpoint `/version` confirme que l'instance interrogée est en V7.2.0.
- La question sur les exclusions de la garantie personnelle ou familiale renvoie encore le message de repli : aucune citation n'est détectée.
- Les documents récupérés contiennent pourtant une source directement pertinente (S3) et une source similaire (S1). S1 combine toutefois les exclusions de deux garanties ; S4 concerne spécifiquement les événements extérieurs et ne doit pas être confondu avec la garantie personnelle/familiale.
- Durées observées : embedding 54,26 ms ; recherche 39,29 ms ; reranking 4 040,10 ms ; génération 91 396,81 ms ; total 95 530,89 ms. La génération représente environ 95,7 % du total pour cet essai isolé.
- Conclusion : la récupération documentaire semble avoir trouvé la matière pertinente, mais la génération/citation reste en échec réel. La cause exacte n'est pas confirmée, car les réponses brutes avant audit ne sont pas disponibles.
- Étape suivante : journaliser de manière contrôlée les deux réponses brutes et le détail de chaque audit, puis corriger sur la base de ces observations. Ne pas conclure à une validation complète avant de nouveaux essais réels.



## V7.2.1 — diagnostic contrôlé de la génération

- Ajout d'un mode de diagnostic activable localement avec la variable d'environnement `RAG_DIAGNOSTICS=1`.
- Lorsque ce mode est activé, la console affiche pour chaque appel Ollama la durée de l'appel, le texte brut généré et le détail de l'audit de citations.
- Le mode est désactivé par défaut afin de ne pas exposer systématiquement dans les journaux les sorties pouvant reprendre la question ou le contenu documentaire.
- Aucun changement n'est apporté au retrieval, au reranking, au prompt fonctionnel ou au mécanisme de repli.
- Cette version est un instrument de diagnostic, pas une correction confirmée du défaut de citation. La validation réelle reste à effectuer avec Ollama.
