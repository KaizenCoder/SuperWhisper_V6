Phase 0 : Finalisation et Validation du MVP (Durée : ≤ 1 journée)
Objectif : Clore la phase en validant le code existant, en corrigeant les bugs et en mesurant les performances.
Go/No-Go : La phase est terminée si les 5 tâches suivantes sont accomplies et que la latence mesurée est < 1.2s.
Tâches :
Créer validate_piper.ps1 : Un script PowerShell pour tester piper.exe en isolation.
Corriger test_tts_handler.py : Remplacer la référence au modèle upmc par siwis.
Valider l'intégration dans run_assistant.py : S'assurer que l'appel tts_handler.speak() est bien présent et fonctionnel dans la boucle principale.
Instrumenter la latence : Ajouter des mesures de temps (time.perf_counter()) dans run_assistant.py pour chronométrer chaque étape (STT, LLM, TTS) et le total.
Mettre à jour la documentation et versionner : Mettre à jour le statut dans les documents et créer un tag Git mvp-p0-validated une fois la validation réussie.
Phase 1 : Rattrapage - Sécurité & Qualité (Durée : 4 Sprints d'une semaine)
Objectif : Résoudre la dette technique critique avant d'ajouter de nouvelles fonctionnalités.
Sprints :
Sprint 1 (Sécurité) :
Objectif : Implémenter l'authentification de base.
KPI de sortie : Endpoints de la future API protégés, tests de validation de token réussis.
Sprint 2 (Tests Unitaires) :
Objectif : Augmenter la confiance dans le code existant.
KPI de sortie : Couverture de tests ≥ 60% sur les modules STT, LLM et TTS.
Sprint 3 (Tests d'Intégration) :
Objectif : Automatiser la validation du pipeline.
KPI de sortie : Pipeline CI/CD sur GitHub Actions est "vert" à chaque push.
Sprint 4 (Robustesse) :
Objectif : Rendre l'application résiliente aux pannes.
KPI de sortie : Un FallbackManager basique est fonctionnel et une démonstration de "crash-test" est réussie.
Phase 2 : Développement des Fonctionnalités Core (Durée : 6 semaines)
(Plan inchangé, axé sur le RAG, le monitoring avancé, et l'intégration Talon).