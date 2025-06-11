# Plan de Développement Détaillé - LUXA

## Phase 0 : Finalisation et Validation du MVP (Durée : ≤ 1 journée - PRIORITÉ ABSOLUE)
* **Objectif :** Clore officiellement la phase en validant la capture audio réelle.
* **Tâches :**
    1.  **Mise à Jour des Dépendances :** Installer `pytest`, `faster-whisper==1.0.0`, `silero-vad==0.4.0`, `sounddevice` et `soundfile`.
    2.  **Créer le Script de Test :** Implémenter `tests/test_realtime_audio_pipeline.py` comme spécifié dans l'avis d'O3.
    3.  **Exécuter et Valider :** Lancer les tests et confirmer que la chaîne `Microphone → VAD → STT` est fonctionnelle.
    4.  **Versionner :** Créer un tag Git `mvp-p0-realaudio-validated`.

## Phase 1 : Rattrapage - Sécurité & Qualité (Durée : 4 Sprints d'une semaine)
* **Objectif :** Résoudre la dette technique avant d'ajouter de nouvelles fonctionnalités.

### Sprint 1 - Sécurité
* **Objectif :** Implémenter l'authentification de base.
* **KPI de sortie :** Endpoints API protégés, tests de validation de token réussis.
* **Livrables :** Module `security/`, middleware FastAPI, suite de tests `tests/test_security.py`.

### Sprint 2 - Tests Unitaires
* **Objectif :** Augmenter la confiance dans le code existant.
* **KPI de sortie :** Couverture de tests ≥ 60% sur les modules STT/LLM/TTS.
* **Livrables :** Suites de tests complètes pour chaque handler, rapports de couverture.

### Sprint 3 - Tests d'Intégration et CI/CD
* **Objectif :** Automatiser la validation du pipeline.
* **KPI de sortie :** Le pipeline GitHub Actions est "vert" à chaque push.
* **Livrables :** Fichier `.github/workflows/ci.yml`, tests d'intégration de bout en bout.

### Sprint 4 - Robustesse
* **Objectif :** Rendre l'application résiliente aux pannes.
* **KPI de sortie :** Le `FallbackManager` est fonctionnel et validé par un "crash-test".
* **Livrables :** `FallbackManager`, `Circuit Breakers` intégrés dans l'orchestrateur.

## Phase 2 : Fonctionnalités Avancées (Post-Phase 1)
* **Monitoring Avancé :** Intégration Prometheus et dashboard Grafana.
* **Retrieval-Augmented Generation (RAG) :** Intégration ChromaDB.
* **Intégration Talon :** Développement d'un prototype de `TalonBridge`.