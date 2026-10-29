# Version 7.5.0 — prototype intégré d'aide à l'analyse des sinistres

## Valeur ajoutée

- Ajout d'un jeu de démonstration de 120 sinistres entièrement fictifs.
- Ajout d'indicateurs descriptifs qui distinguent les dossiers acceptés, refusés et en attente.
- Exclusion explicite des dossiers `Pending` du dénominateur du taux d'acceptation et de l'apprentissage du modèle.
- Ajout d'une régression logistique interprétable, évaluée sur un jeu de test tenu à part avant le réentraînement de démonstration.
- Ajout d'un tableau de bord `/claims` et de routes API dédiées aux indicateurs, aux dossiers, aux métriques et à l'analyse illustrative d'un dossier.
- Ajout d'avertissements visibles : données fictives, score exploratoire et validation humaine obligatoire.
- Conservation du RAG, de la recherche hybride, du contrôle des citations et des contrôles opérationnels introduits précédemment.

## Vérifications

- Compilation syntaxique réussie.
- Suite automatisée exécutée : 40 tests réussis sur 40 dans l’environnement de développement.
- Les tests simulés ne remplacent pas la recette de bout en bout avec Ollama et les modèles réels.

## Limites

Le jeu de données est synthétique et ne représente pas un portefeuille réel. Les métriques du modèle ne doivent pas être interprétées comme une estimation de performance en production. Le système ne prend aucune décision automatique et ne remplace pas l'analyse du contrat, des pièces et des règles de gestion par un gestionnaire habilité.
