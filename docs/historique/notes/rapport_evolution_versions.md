# Évolution du projet — démarche et résultats

## 1. Objectif

Je développe une mini-application RAG pour rechercher des documents d'assurance et produire une réponse en français à partir des documents récupérés. Le projet combine une recherche initiale par embeddings, un reclassement par CrossEncoder et une génération de réponse par Ollama. Je distingue les mesures du classement des documents des vérifications portant sur la qualité des réponses générées.

## 2. Démarche

Je pars d'un constat, formule une hypothèse d'amélioration, applique un changement ciblé, puis vérifie ce changement avec des tests adaptés. Je distingue les tests unitaires simulés des essais de bout en bout avec les vrais modèles. Un test unitaire réussi confirme un comportement dans un scénario contrôlé ; il ne prouve pas à lui seul qu'une réponse d'assurance est correcte.

## 3. Évolution mesurée de `src/app.py`

Le nombre de lignes a été mesuré directement dans les fichiers des archives reçues, en comptant les lignes séparées par des retours à la ligne :

| Version | Lignes de `src/app.py` | Évolution par rapport à la version précédente |
|---|---:|---:|
| V3 | 301 | Base de comparaison |
| V4 | 311 | +10 |
| V5 | 351 | +40 |
| V6 | 355 | +4 |
| V7 | 377 | +22 |

Le nombre de lignes est un indicateur descriptif, pas une mesure directe de qualité. Chaque ajout doit être justifié par une fonctionnalité utile, une meilleure observabilité ou un contrôle vérifiable.

## 4. Décisions techniques des versions 3 à 6

### V3 — validation et erreurs

La version de travail comportait la validation des questions et une gestion plus explicite des erreurs de génération. Les tests couvraient notamment les limites de longueur et des réponses Ollama non exploitables.

### V4 — traçabilité des sources et séparation des scores

La V4 a ajouté des identifiants de source comme `[S1]` aux documents transmis au générateur. L'API distingue le score du reranker de la similarité des embeddings, tout en conservant le champ historique `score` pour la compatibilité. Le code vérifie également que la réponse d'Ollama contient du texte exploitable.

### V5 — contrôle formel des identifiants et chemins robustes

La V5 inspecte les identifiants de source présents dans la réponse. L'API distingue les identifiants connus, les identifiants inconnus et les sources disponibles mais non citées. Le contrôle est syntaxique : une citation reconnue ne prouve pas que le document justifie l'affirmation. Les chemins du CSV et de la page d'accueil sont calculés à partir du dossier du projet.

### V6 — lisibilité des citations et des scores

La V6 affiche dans l'interface les sources reconnues, les identifiants inconnus et les sources disponibles mais non citées. Elle précise que le score brut du CrossEncoder n'est pas une probabilité et que la similarité cosinus ne garantit pas la véracité d'une réponse.

## 5. V7 — rendre le pipeline plus observable

### Constat

Avant la V7, la réponse de l'API ne donnait pas de mesure de durée par étape. Pendant la campagne de validation, il aurait été difficile d'identifier si le temps était surtout consommé par l'encodage, la recherche, le reranking ou la génération.

### Hypothèse et décision

J'ai supposé que des mesures par étape faciliteraient l'analyse des performances et la documentation des essais de bout en bout. J'ai donc instrumenté les principales étapes sans changer les modèles ni la stratégie de classement.

### Changements

- `src/app.py` renvoie les durées `embedding_ms`, `retrieval_ms`, `reranking_ms`, `generation_ms` et `total_ms`.
- La réponse comprend également le nombre de documents du corpus, de candidats examinés et de résultats renvoyés.
- `static/index.html` affiche les durées mesurées pour la requête courante et précise qu'elles dépendent de la machine et de sa charge.
- Les tests vérifient la présence, le type et la non-négativité des métriques, ainsi que l'existence de leur affichage dans l'interface.

### Résultats automatisés

La suite de tests compte 19 tests ; les 19 ont réussi dans l'environnement de vérification. La compilation syntaxique des fichiers Python a également réussi. Les tests utilisent des doubles pour les modèles d'embeddings, le CrossEncoder et Ollama. Les durées de ces tests ne représentent donc pas les performances réelles de l'application.

## 6. Résultats disponibles sur le retrieval

L'évaluation existante contient 201 questions. Les résultats rapportés sont :

| Métrique | TF-IDF | Embeddings |
|---|---:|---:|
| Recall@1 | 29,85 % | 37,81 % |
| Recall@5 | 62,19 % | 67,16 % |
| MRR | 0,4352 | 0,5051 |

Pour le reranking, la cible attendue se trouve dans le Top-20 initial pour 167 questions sur 201, soit 83,08 %. Pour 34 questions (16,92 %), la cible est absente de la sélection initiale. Après reranking, le Hit Rate@5 rapporté est de 76,12 % et le MRR de 0,6331.

Ces métriques évaluent le classement des documents. Elles ne démontrent pas que les réponses générées par Ollama sont exactes ni que les citations justifient les affirmations.

## 7. Campagne de validation à effectuer

Les tests finaux restent à effectuer avec les vrais modèles et un serveur Ollama actif. La campagne devra comprendre :

1. une question dont la réponse figure clairement dans le corpus ;
2. une question nécessitant plusieurs documents ;
3. une question dont la réponse est absente du corpus ;
4. une vérification manuelle de la correspondance entre les affirmations et les sources citées ;
5. un cas de citation manquante et un identifiant inconnu ;
6. une question vide et une question trop longue ;
7. une indisponibilité d'Ollama et une réponse Ollama invalide ;
8. un lancement depuis un répertoire différent de la racine ;
9. la vérification de l'interface et de la cohérence des scores ;
10. la collecte des durées réelles du pipeline sur plusieurs questions, dans un environnement documenté.

Pour chaque scénario, je noterai la question, le résultat attendu, le résultat observé, le statut et la correction éventuelle. Je ne déclarerai pas le projet validé avant l'exécution de ces scénarios.

## 8. Limites et prochaines étapes

Le contrôle des citations reste formel et ne juge pas le fond. Les scores ne sont ni des probabilités ni des garanties de pertinence factuelle. Les mesures de durée de la V7 sont des indicateurs par requête ; elles ne remplacent pas une campagne de performance contrôlée. La prochaine étape consiste à lancer l'application avec les modèles réels, recueillir les résultats, corriger les problèmes détectés puis intégrer les observations dans le rapport. Après validation, je pourrai stabiliser une version et préparer le déploiement pour le partage avec Keyzan.


## 9. Correctif V7.1 — réponse insuffisamment référencée

### Constat issu des essais réels

Pendant le deuxième essai de bout en bout, j'ai posé une question ciblée sur les exclusions de la garantie optionnelle de revente suite à un événement personnel ou familial. Les documents les plus pertinents figuraient dans les cinq résultats, et la réponse reprenait le contenu du document dédié, mais le contrôle affichait « aucune citation détectée ». Ce résultat montre que le prompt initial ne suffit pas à garantir la présence de références dans la réponse.

### Hypothèse et décision

J'ai retenu l'hypothèse qu'une consigne de correction ciblée pourrait récupérer une réponse correctement référencée sans modifier la sélection documentaire. J'ai donc ajouté une seule tentative de révision lorsque la première réponse n'inclut pas de citation disponible ou contient une référence inconnue. Si la seconde réponse échoue aussi au contrôle formel, l'application affiche un message de repli prudent plutôt que le texte généré non référencé.

### Changements et portée

Le changement est limité à la génération de la réponse. Les embeddings, le classement initial, le reranker, le nombre de candidats, les cinq documents transmis, l'interface générale et les métriques de durée sont conservés. La logique n'affirme pas vérifier la validité sémantique des citations ; elle contrôle uniquement leur présence et leur appartenance aux identifiants disponibles.

### Résultats automatisés

Après ce correctif, j'ai exécuté la suite de tests automatisés dans l'environnement de travail : **22 tests réussis sur 22**. Les deux tests ajoutés vérifient (1) qu'une réponse sans citations entraîne une seconde tentative et qu'une réponse révisée avec une référence disponible est retenue, et (2) qu'un message de repli est utilisé si la seconde réponse n'est toujours pas référencée. Les appels Ollama de ces deux tests sont simulés. La compilation syntaxique est vérifiée séparément.

### Conséquence attendue et limite

Le comportement attendu est une meilleure traçabilité formelle. La conséquence possible est une latence accrue, car une réponse qui échoue au premier contrôle peut déclencher un deuxième appel au modèle. Il reste à exécuter de nouveau les deux questions réelles dans l'interface, puis à vérifier manuellement que les affirmations correspondent aux extraits cités. Les tests automatisés ne suffisent pas à conclure que la V7.1 est validée de bout en bout.

## 10. Essai de bout en bout V7.2.0 — absence persistante de citations

### Question et résultat observé

J'ai relancé dans l'interface la question ciblée : « Dans quels cas la garantie optionnelle de revente suite à un événement personnel ou familial ne s'applique-t-elle pas ? ». L'instance a été identifiée comme V7.2.0 via `GET /version`. Malgré cette vérification, l'application a encore affiché le message de repli : « Je ne peux pas fournir une réponse suffisamment traçable à partir des documents récupérés ». Le contrôle a signalé aucune citation détectée, alors que les sources S1 à S5 étaient disponibles.

### Éléments documentaires observés

Les documents récupérés contiennent bien la matière nécessaire pour répondre :

- **S3** correspond directement à la question et énumère les exclusions de la garantie personnelle ou familiale : suicide ou tentative, accident aérien avec exception pour le simple passager d'un vol commercial, guerre, terrorisme/sabotage/attentat dans les conditions précisées, accident lié à l'ivresse ou à certains produits, certains sports motorisés, dépendance située à une autre adresse, biens immobiliers classés ou assimilés et mobile-home.
- **S1** contient une liste largement similaire, mais présente aussi les exclusions relatives à la garantie de revente suite à un événement extérieur. Il faut donc distinguer les deux garanties pour éviter de mélanger leurs exclusions.
- **S2** décrit principalement les événements personnels ou familiaux pouvant déclencher la garantie, plutôt que la liste des exclusions.
- **S4** concerne la garantie liée aux événements extérieurs et ne doit pas être utilisé comme source des exclusions propres à la garantie personnelle ou familiale.
- **S5** porte sur la garde d'enfants et n'est pas pertinent pour cette question.

La présence de S3 parmi les résultats confirme que la recherche documentaire a récupéré une source directement pertinente pour cet essai. Elle ne prouve toutefois pas, à elle seule, que toute la chaîne de génération et de citation fonctionne correctement.

### Mesures de latence relevées

| Étape | Durée observée |
|---|---:|
| Embedding | 54,26 ms |
| Recherche | 39,29 ms |
| Reranking | 4 040,10 ms |
| Génération | 91 396,81 ms |
| Total | 95 530,89 ms |

Ces valeurs sont celles d'une seule requête et varient selon la machine, la charge et l'exécution des modèles. La génération représente environ 95,7 % du temps total mesuré. Le reranking prend environ 4 secondes, mais n'explique pas la majeure partie de l'attente. La durée totale est compatible avec un coût important de génération, possiblement aggravé par une seconde tentative ; les métriques agrégées affichées ne permettent pas de confirmer combien de temps chaque appel Ollama a consommé.

### Analyse et limites du diagnostic

Le résultat montre que le correctif V7.2.0 n'a pas encore résolu le problème observé en conditions réelles : la réponse finale est toujours rejetée faute de citation détectée. Il ne permet pas de conclure que la récupération documentaire a échoué, puisque S3 est présent et directement pertinent.

L'hypothèse de travail est que le défaut se situe désormais dans la sortie du modèle, dans la façon dont le prompt de révision est interprété, ou dans le format exact des citations produites et reconnues par l'audit. Ces possibilités restent à vérifier : aucune réponse brute d'Ollama avant l'audit n'a été conservée dans les éléments de cet essai. Il serait donc prématuré d'attribuer définitivement la cause à l'une d'elles.

Le contrôle de citations est volontairement limité : il vérifie la présence d'identifiants reconnus et leur appartenance aux sources disponibles, mais ne vérifie pas que chaque affirmation est réellement étayée par la source citée. Le repli évite d'afficher une réponse dont la traçabilité formelle échoue ; en revanche, dans ce cas, il prive l'utilisateur d'une réponse alors que les documents semblent contenir les éléments nécessaires.

### Actions recommandées

1. Ajouter une journalisation contrôlée de la réponse brute du premier appel Ollama, avant l'audit, et de la réponse brute de la seconde tentative, en veillant à ne pas exposer d'informations sensibles dans les journaux.
2. Enregistrer pour chaque tentative le résultat de l'audit : citations reconnues, identifiants inconnus et sources disponibles non citées. Cela permettra de distinguer une absence réelle de citations d'un problème de format ou d'analyse.
3. Relancer uniquement la question ciblée et comparer les deux sorties avant de modifier encore le prompt. Ne pas changer simultanément la recherche, le reranking et la génération, afin de préserver l'identifiabilité de la cause.
4. Après correction, tester à nouveau cette question et la question générale de prise en charge d'un sinistre. Pour cette dernière, contrôler particulièrement le risque de confondre les exclusions de S1/S3 avec les dispositions relatives aux événements extérieurs de S4.
5. Mesurer séparément le temps de chaque appel Ollama et le nombre d'appels réellement exécutés. Réduire la latence seulement après avoir identifié la part respective de la première génération et de la tentative de révision.

### Statut de validation

**V7.2.0 : validation de bout en bout non acquise pour la génération avec citations.** L'endpoint de version confirme que l'instance interrogée est V7.2.0, mais l'essai réel échoue encore au contrôle des citations. Les tests automatisés antérieurs vérifient le comportement programmé avec des appels simulés ; ils ne garantissent pas que le modèle réel respecte la consigne. La prochaine étape est un diagnostic instrumenté des deux sorties brutes, puis une correction ciblée et une nouvelle validation réelle.



## 11. V7.2.1 — instrumenter les deux appels de génération

### Objectif

À la suite de l'échec persistant observé en V7.2.0, je ne modifie pas encore la stratégie de recherche ni les consignes de génération. L'objectif de cette version est de collecter les éléments manquants pour établir un diagnostic fondé sur les sorties réelles du modèle.

### Modifications

La variable d'environnement `RAG_DIAGNOSTICS=1` active une journalisation locale des éléments suivants pour chaque appel Ollama : durée individuelle, sortie textuelle brute et résultat détaillé de l'audit des citations. Le mode est désactivé par défaut, car les sorties peuvent contenir des éléments de la question et du corpus. Les prompts complets et le contenu complet des documents ne sont pas journalisés par ce mécanisme. La recherche, le reranking, le prompt fonctionnel et le repli prudent ne sont pas modifiés. L'endpoint `/version` expose désormais la version `7.2.1`.

### Vérifications automatisées

La suite automatisée a été exécutée sur cette archive après modification : **25 tests réussis sur 25**. La compilation syntaxique de `src/app.py` et du fichier de tests a également réussi. Le nouveau test vérifie que le mode de diagnostic affiche les deux sorties simulées, les audits et les durées des deux appels. Ce test vérifie le mécanisme de journalisation avec des réponses simulées ; il ne constitue pas un essai réel d'Ollama.

### Procédure de diagnostic réel sous Windows PowerShell

Depuis le dossier du projet V7.2.1, arrêter d'abord l'ancienne instance si elle occupe le port choisi, puis lancer :

```powershell
$env:RAG_DIAGNOSTICS = "1"
python -m uvicorn src.app:app --host 127.0.0.1 --port 8001
```

Dans l'interface, poser une seule fois la question ciblée sur les exclusions de la garantie optionnelle de revente suite à un événement personnel ou familial. Copier ensuite les lignes de console commençant par `[RAG_DIAGNOSTICS]` pour la tentative 1 et la tentative 2, y compris leurs audits et durées. Avant de partager ces journaux, vérifier qu'ils ne contiennent pas d'informations personnelles. Une fois le diagnostic terminé, arrêter le serveur et relancer l'application sans la variable, ou exécuter `Remove-Item Env:RAG_DIAGNOSTICS` avant de la relancer.

### Statut

**Diagnostic instrumenté prêt ; cause racine non encore établie.** Aucun résultat réel avec cette version n'est revendiqué à ce stade. La prochaine décision dépendra des sorties et audits observés : il faudra distinguer une réponse sans citation, une sortie vide/mal formée ou un format de référence non reconnu avant de modifier le comportement.


## 12. V7.3.0 — enrichir la présélection avec les termes métier

### Constat observé

Lors d'un essai sur la question « Comment fonctionne la franchise absolue d'une assurance habitation ? », la recherche a présenté cinq documents qui ne contenaient pas la définition attendue. Le corpus comporte pourtant un document intitulé « Comment fonctionne la franchise d’une assurance habitation ? ». Ce constat montre que la présélection exclusivement sémantique peut écarter une source métier utile avant même que le CrossEncoder ne la classe.

### Décision technique

La V7.3.0 conserve les embeddings et le CrossEncoder, puis complète la présélection avec un signal lexical fondé sur les termes significatifs de la question, leur présence dans le titre et leur présence dans le contenu. La sélection combine les rangs sémantiques et lexicaux avec une fusion de rangs réciproques. Cette méthode évite d'additionner directement des scores qui ne partagent pas la même échelle. Le score lexical est exposé dans les résultats à titre indicatif et ne doit pas être interprété comme une probabilité ni comme une preuve de pertinence factuelle.

### Valeur ajoutée métier

Cette amélioration vise à préserver les documents qui contiennent des expressions exactes telles que « franchise absolue », « franchise relative », « plafond de garantie » ou « exclusion », même lorsque la formulation de la question est courte ou ambiguë. Le CrossEncoder conserve ensuite son rôle de classement final. La génération de réponse n'est pas considérée comme validée par la seule présence d'une citation : la qualité du raisonnement et la fidélité aux passages restent des critères distincts.

### Vérifications automatisées

Deux tests supplémentaires vérifient que le document contenant le terme « franchise » reste parmi les candidats prioritaires dans un scénario contrôlé et que le score lexical favorise un titre correspondant précisément à la question. Ces tests complètent la suite existante. Ils valident la logique de sélection sur des données contrôlées, mais ils ne remplacent pas un essai de bout en bout avec les modèles réels et Ollama.

### Limites et prochaine vérification

La fusion lexicale peut favoriser un document qui reprend les mêmes mots sans répondre à toute la question. Le reranking et l'examen du passage restent donc nécessaires. La prochaine validation réelle doit utiliser la question sur la franchise absolue, puis vérifier que la source de définition figure dans les cinq résultats, que les exemples numériques restent exacts et que la réponse distingue bien la franchise absolue de la franchise relative. Les temps d'exécution réels devront être comparés avant de conclure que la performance s'est améliorée.

### Statut de validation

La V7.3.0 constitue une amélioration ciblée de la sélection des candidats. La validation automatisée doit réussir avant livraison. La pertinence réelle, la fidélité factuelle et les performances avec les modèles installés restent à confirmer dans un essai local contrôlé.


## 13. V7.4.0 — préparation opérationnelle et fusion finale des rangs

### Objectif de livraison

Cette version prépare une recette contrôlée par Keyzan. Elle ne se limite pas à faire démarrer l'application : elle rend la configuration plus facile à adapter, expose des contrôles de disponibilité et formalise les essais qui doivent être exécutés avant un partage.

### Amélioration de la recherche

La version précédente fusionnait les rangs sémantiques et lexicaux pour sélectionner les candidats, puis elle laissait le CrossEncoder déterminer seul les cinq sources présentées. La V7.4.0 ajoute une fusion finale entre le rang du CrossEncoder et le rang lexical. Cette étape peut préserver une source qui reprend exactement une notion métier importante, comme la franchise absolue, sans additionner directement des scores dont les échelles sont différentes. L'interface présente le score hybride comme un indicateur de classement, et non comme une probabilité ou une preuve de vérité.

### Configuration et contrôles opérationnels

Les variables `OLLAMA_URL`, `OLLAMA_MODEL` et `OLLAMA_TIMEOUT_SECONDS` permettent d'adapter l'application à l'environnement de test sans modifier le code source. La route `/health` confirme que le processus API répond. La route `/ready` vérifie les fichiers nécessaires, le chargement des modèles, l'accessibilité d'Ollama et la présence du modèle configuré. Une disponibilité technique ne garantit toutefois ni la pertinence des sources ni l'exactitude des réponses.

### Sécurité et partage

L'application ne possède pas d'authentification applicative intégrée. La livraison recommande donc un test local ou un réseau privé contrôlé. L'application ne doit pas être exposée directement à Internet sans dispositif d'accès approuvé. Les premiers essais ne doivent pas utiliser de données personnelles ou de dossiers réels de sinistres.

### Vérifications réalisées dans l'environnement de développement

La suite automatisée a été exécutée après les modifications : **34 tests réussis sur 34**. La compilation syntaxique de l'application et du fichier de tests a également réussi. Les tests de disponibilité utilisent un faux service Ollama ; ils valident le traitement des états prêt et non prêt, mais ils ne démontrent pas qu'Ollama réel fonctionne sur la machine de livraison.

### Vérifications restant à effectuer avant validation finale

Il reste à lancer l'application avec les vrais modèles et le service Ollama actif. Le test de franchise absolue doit confirmer que le document de définition figure parmi les cinq sources et que chaque affirmation importante de la réponse est soutenue par le passage cité. Une question hors corpus doit également provoquer une réponse prudente. Enfin, il faut relever les durées sur plusieurs questions, vérifier le démarrage depuis un autre répertoire et faire compléter la fiche de recette par Keyzan.

### Statut

La V7.4.0 est préparée pour une recette contrôlée. Les tests automatisés sont réussis dans l'environnement de développement, mais la validation de bout en bout avec Ollama réel, la qualité métier et les performances sur la machine cible restent à confirmer.


## 14. V7.5.0 — prototype d'aide à l'analyse des sinistres

### Alignement avec l'objectif métier

Le projet vise un système d'aide à l'analyse des sinistres, et non un moteur qui décide automatiquement d'accepter ou de refuser une indemnisation. La V7.5.0 conserve l'assistant RAG et ajoute un tableau de bord qui illustre la complémentarité entre les données d'un dossier, les indicateurs de portefeuille, un score exploratoire et les informations contractuelles qui restent à rechercher dans le RAG.

### Jeu de données fictif et indicateurs

Le fichier `Data/synthetic_claims.csv` contient 120 dossiers fictifs avec le type de sinistre, le montant demandé, l'ancienneté client, l'historique des sinistres, le canal de déclaration, la complétude du dossier, le statut et la date de déclaration. Le tableau de bord calcule le nombre de dossiers acceptés, refusés et en attente. Le taux d'acceptation exclut explicitement les dossiers `Pending` du dénominateur, car leur décision finale n'est pas connue. L'âge du dossier est calculé à partir de la date de déclaration afin de mettre en évidence les dossiers qui restent en attente.

### Modèle de démonstration

Le module `src/claims_analytics.py` entraîne une régression logistique avec encodage des variables catégorielles et standardisation des variables numériques. L'évaluation utilise un découpage stratifié entre entraînement et test. Le modèle exclut les dossiers `Pending` de l'apprentissage et présente plusieurs métriques, dont l'exactitude équilibrée, le F1-score et l'AUC lorsque cette dernière est calculable. Le modèle est ensuite réentraîné sur tous les dossiers clos pour illustrer un score sur un dossier de démonstration.

### Limites et gouvernance

Les données sont synthétiques et les résultats ne permettent pas d'évaluer un vrai portefeuille. Les probabilités sont des démonstrations techniques et ne doivent pas être interprétées comme des probabilités calibrées pour l'activité d'un assureur. Le score n'intègre pas encore l'analyse RAG d'un contrat spécifique, les pièces justificatives ni les règles de gestion métier. Une décision humaine reste obligatoire. Le projet ne doit pas utiliser de données personnelles ou de dossiers réels sans une gouvernance adaptée.

### Interface et routes ajoutées

La route `/claims` sert un tableau de bord qui permet de consulter les indicateurs, de parcourir les dossiers fictifs et d'afficher une analyse exploratoire. Les routes `/api/claims/summary`, `/api/claims/cases`, `/api/claims/model` et `/api/claims/{claim_id}/analysis` exposent les données correspondantes. La page d'accueil du RAG contient un lien vers le tableau de bord, et le tableau de bord permet de revenir à la recherche documentaire.

### Vérifications automatisées

La suite automatisée a été exécutée après les modifications : **40 tests réussis sur 40**. Les tests vérifient que le taux d'acceptation exclut les dossiers en attente, que les données exposées restent fictives, que les scores sont bornés entre zéro et un, que le modèle n'utilise pas les dossiers Pending et que les routes analytiques sont enregistrées. Ces tests fonctionnels ne remplacent pas une validation métier indépendante ni une évaluation sur des données réelles autorisées.

### Statut

La V7.5.0 constitue un prototype plus proche de la vision de système d'aide à la décision décrite dans le dossier de conception. Le RAG, l'analytique et le modèle sont réunis dans une démonstration cohérente, mais la validation de bout en bout avec Ollama réel, la qualité des réponses contractuelles et la validation actuarielle du modèle restent à effectuer avant tout usage opérationnel.


## V15 — audit de la récupération et enrichissement de l'analyse technique

La V15 ajoute un audit technique documenté de la V14 avant toute modification fonctionnelle du moteur RAG. Cette étape établit une base de comparaison explicite, décrit l'architecture réellement présente dans le code et distingue les résultats mesurés des hypothèses qui restent à vérifier.

L'audit confirme que la suite automatisée compte 40 tests et qu'ils réussissent dans l'environnement de vérification utilisé. Ce résultat valide les comportements couverts par les tests, mais il ne constitue pas une preuve autonome de la pertinence des documents récupérés ni de la fidélité des réponses produites par un serveur Ollama réel.

Les fichiers historiques d'évaluation contiennent 201 questions. Le document attendu apparaît dans le Top 5 dans 67,16 % des cas pour la colonne d'évaluation par embeddings et dans 76,12 % des cas pour la colonne d'optimisation enregistrée, soit une variation absolue de 8,96 points de pourcentage. Le document attendu reste absent du Top 20 dans 34 cas sur 201. Ces chiffres sont rapportés comme des résultats historiques issus des CSV du projet ; ils ne doivent pas être présentés comme une nouvelle mesure de bout en bout de la V14 tant que le protocole et les paramètres ne sont pas reproduits.

Le nouveau document `docs/audit_technique_v14_et_plan_v15.md` explique la similarité cosinus, le signal lexical, la Reciprocal Rank Fusion et le rôle du CrossEncoder. Il décrit aussi les limites des scores, la différence entre une citation syntaxiquement valide et une affirmation réellement justifiée, ainsi que le protocole recommandé pour mesurer Recall@k, Precision@k, MRR, nDCG, la fidélité factuelle et les temps de chaque étape.

Cette étape est volontairement additive : elle n'enlève aucune fonctionnalité de l'application et ne modifie pas encore le comportement du pipeline. La prochaine modification du moteur devra être guidée par les cas d'échec documentés et validée par une comparaison reproductible sur un jeu de questions annotées.

## Version V16 — correction de la fusion lexicale en l'absence de correspondance

### Contexte et objectif

Après l'audit de la V15, j'ai poursuivi l'examen du classement hybride. J'ai identifié un défaut conceptuel dans la manière dont le rang lexical contribuait au score de fusion. La fonction de classement attribuait un rang à chaque document, y compris aux documents qui ne partageaient aucun terme significatif avec la question. Ce rang était ensuite utilisé dans la fusion finale comme si chaque document disposait d'un signal lexical informatif.

Cette situation pouvait introduire un biais artificiel. Un rang lexical existe toujours dès lors que l'on trie une liste, mais son existence ne prouve pas qu'un document correspond lexicalement à la requête. Lorsque plusieurs documents n'obtiennent aucune correspondance, leur ordre peut dépendre de règles de départage sans rapport avec le sens de la question. Il est donc préférable de ne pas accorder de contribution lexicale positive à un document dont le score de correspondance est nul.

### Modification réalisée

J'ai ajouté la fonction `_rank_fusion_score`, qui calcule la contribution du classement principal et ajoute la contribution lexicale uniquement lorsque le score lexical est strictement positif. J'ai appliqué cette règle à la fusion des candidats et au classement final après le CrossEncoder. J'ai également conservé cette définition dans le champ `hybrid_rank_score` renvoyé par l'API, afin que le score exposé corresponde à la règle de classement effectivement appliquée.

La règle utilise toujours une fusion réciproque de rangs. Pour un document de rang principal \(r_p\), de rang lexical \(r_l\), un score lexical \(s_l\) et une constante \(k\), le score devient :

\[
H(d)=\frac{1}{k+r_p(d)}+\mathbb{1}[s_l(d)>0]\frac{1}{k+r_l(d)}.
\]

L'indicatrice \(\mathbb{1}[s_l(d)>0]\) vaut un lorsque le document partage au moins un terme retenu par le calcul lexical et zéro dans le cas contraire. Cette définition ne transforme pas le score en probabilité. Elle évite uniquement qu'un classement lexical sans correspondance ajoute artificiellement un bonus au document.

### Portée et précautions

Cette modification est ciblée. Elle ne remplace pas les embeddings, le CrossEncoder, le corpus documentaire ou le modèle Ollama. Elle ne prétend pas résoudre les erreurs de compréhension sémantique, les variantes morphologiques, les synonymes, les négations ou les cas dans lesquels le document pertinent ne figure pas parmi les candidats. Elle ne constitue pas non plus une preuve de hausse du Recall@5 : cette hausse doit être mesurée sur un jeu de questions annotées en comparant les mêmes questions, les mêmes documents et les mêmes paramètres avant et après le changement.

### Vérifications automatisées

J'ai ajouté deux tests unitaires ciblés. Le premier vérifie qu'un document sans correspondance lexicale ne reçoit aucune contribution lexicale, même si un rang lexical lui a été attribué. Le second vérifie qu'un document ayant une correspondance positive conserve les deux contributions de rang. Après la modification, la suite automatisée compte 42 tests réussis. La compilation des modules Python se termine également sans erreur.

Les tests utilisent des modèles simulés et vérifient la formule de fusion. Ils ne mesurent pas la qualité du modèle d'embeddings ni celle du CrossEncoder sur le corpus réel. Ils ne constituent pas un essai de bout en bout avec Ollama. Je ne présente donc aucun gain de qualité ou de latence comme établi par cette seule vérification.

### Suivi de la taille des fichiers

Le fichier `src/app.py` est passé de 726 à 754 lignes, soit une augmentation de 28 lignes. Le fichier `tests/test_app_validation.py` est passé de 497 à 521 lignes, soit une augmentation de 24 lignes. Le rapport d'évolution conserve son contenu précédent et reçoit cette nouvelle section. Ces évolutions respectent la règle du projet qui consiste à enrichir les fichiers sans réduire leur nombre de lignes.

### Prochaine évaluation recommandée

La prochaine étape consiste à rejouer les 201 questions de référence avec la version précédente et la V16. Je comparerai le Recall@1, le Recall@3, le Recall@5, le Recall@20 et le MRR. J'inspecterai séparément les questions améliorées, les questions dégradées et les questions pour lesquelles le document de référence reste absent du Top 20. Je mesurerai ensuite la latence de récupération et du reranking sur plusieurs exécutions, en documentant le matériel et les modèles utilisés. Enfin, je contrôlerai manuellement la fidélité des réponses et des citations pour les questions représentatives du domaine de l'assurance.

### Statut

La V16 corrige une incohérence de la formule de fusion et dispose de tests unitaires qui vérifient le comportement attendu. La validation fonctionnelle de la formule est acquise dans l'environnement de tests. La validation de la pertinence réelle du classement et la validation de bout en bout avec les modèles réels restent à effectuer.

## Évaluation complémentaire — référence lexicale mesurée après la V16

J'ai exécuté le script de référence TF-IDF sur les 201 lignes de `Data/insurance-fr.csv`, en utilisant chaque titre comme question et le document de la même ligne comme référence attendue. Cette exécution a donné un Recall@1 de 29,85 % (60/201), un Recall@3 de 52,24 % (105/201), un Recall@5 de 62,19 % (125/201), un Recall@20 de 77,11 % (155/201) et un MRR de 0,4352. Les rangs et les premiers résultats par question sont conservés dans `Data/tfidf_reference_evaluation_v16.csv`, et le protocole est décrit dans `docs/evaluation_retrieval_mesure_v16.md`.

Ces valeurs mesurent uniquement une référence lexicale TF-IDF. Elles ne mesurent pas le pipeline hybride de production de la V16, car `sentence_transformers` n'était pas disponible dans l'environnement utilisé pour cette exécution. Elles ne démontrent donc aucun gain ou aucune régression causé par le correctif de fusion. Les valeurs historiques de 67,16 % et 76,12 % de présence dans le Top 5 restent identifiées comme des résultats consignés antérieurement dans le CSV historique, et non comme une comparaison contrôlée de la V16. La validation avant/après sur les mêmes modèles et paramètres demeure nécessaire.

## Version de paquet V17 — traçabilité de la référence lexicale

La V17 enrichit le paquet de livraison sans modifier le comportement applicatif de la V16. Elle ajoute un rapport de mesure TF-IDF et un CSV contenant les résultats question par question. La suite reste à 42 tests réussis, et la compilation Python réussit. Cette évolution améliore la traçabilité expérimentale ; elle ne doit pas être interprétée comme une preuve d'amélioration du pipeline hybride.

## V18 — expérimentation SVD/LSA comparative (hors production)

J'ai ajouté un script d'évaluation reproductible pour comparer TF-IDF, TF-IDF suivi d'une SVD tronquée (LSA) et des fusions de classements. Les embeddings peuvent également être inclus avec l'option `--with-embeddings` lorsque le modèle est disponible. L'expérience est volontairement séparée de l'application de production afin de mesurer d'abord l'effet de la méthode avant de modifier le RAG actif.

Sur les 201 lignes du corpus, la fusion RRF entre TF-IDF et SVD à 50 composantes atteint Recall@5 = 65,67 %, contre 63,68 % pour TF-IDF dans ce même script. SVD seule ne gagne pas uniformément. Ces valeurs reposent sur un proxy où le titre sert de requête et le document de la même ligne est la cible attendue ; elles ne constituent donc pas une mesure de pertinence métier validée. La comparaison aux embeddings n'a pas été exécutée dans cette campagne et reste à faire avec les poids réels du modèle. Le détail et les limites figurent dans `docs/evaluation_tfidf_svd_embeddings_v18.md`.
