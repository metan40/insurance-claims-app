# Version 7.4.0 — préparation de livraison contrôlée

## Améliorations

- La sélection finale fusionne désormais le rang du CrossEncoder et le rang lexical afin de mieux préserver les termes métier explicites.
- Les paramètres d'Ollama sont configurables par variables d'environnement.
- Le délai maximal d'appel Ollama est configurable et conserve une valeur par défaut prudente.
- La route `/health` distingue la disponibilité du processus de la qualité métier des réponses.
- La route `/ready` vérifie le corpus, l'interface, les modèles chargés, l'accès à Ollama et la présence du modèle configuré.
- Le dossier de livraison explique le démarrage, les limites de sécurité et les scénarios de recette à effectuer avec Keyzan.

## Limites connues

- Les scores et la fusion hybride améliorent le classement, mais ne prouvent pas qu'un passage justifie une réponse.
- La vérification des citations contrôle encore les identifiants et ne réalise pas de validation sémantique de chaque affirmation.
- Les tests automatisés simulant les modèles ne remplacent pas les essais avec Ollama réel.
- L'application ne possède pas d'authentification intégrée. Elle doit rester locale ou sur un réseau privé contrôlé tant qu'un dispositif d'accès adapté n'est pas mis en place.
