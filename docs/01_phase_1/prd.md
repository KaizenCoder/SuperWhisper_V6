# Product Requirements Document (PRD) - SuperWhisper_V6 (LUXA)
**Version :** 1.1
**Date :** 10 juin 2025

## 1. Vue d'Ensemble
SuperWhisper_V6 (LUXA) est une application de bureau Python conçue pour être un assistant vocal intelligent, 100% local et privé. Son objectif est de fournir une expérience voix-à-voix complète et naturelle (STT → LLM → TTS) sans aucune dépendance à des services cloud, garantissant ainsi une confidentialité totale et une faible latence. Le public cible est constitué d'utilisateurs finaux et de technophiles recherchant une solution souveraine.

## 2. Objectifs du Projet
- **Phase 0 (MVP) :** Valider un pipeline voix-à-voix fonctionnel avec des performances mesurables.
- **Phase 1 (Qualité) :** Adresser la dette technique en implémentant la sécurité, les tests et la robustesse.
- **Phase 2 (Fonctionnalités) :** Étendre les capacités avec le RAG, le monitoring avancé et d'autres intégrations.

## 3. Périmètre et Fonctionnalités

### Objectifs de la Phase 0 (En Cours de Finalisation)
1.  **Pipeline Voix-à-Voix Fonctionnel :** Prouver que la chaîne complète `Capture Audio → STT → LLM → TTS → Sortie Audio` est opérationnelle.
2.  **Validation des Composants Clés :**
    - **STT :** Transcription via `insanely-fast-whisper`.
    - **LLM :** Inférence via `llama-cpp-python`.
    - **TTS :** Synthèse via l'exécutable `piper.exe`.
3.  **Instrumentation des Performances :** Mesurer la latence de bout en bout pour établir une baseline.

### Fonctionnalités Prioritaires (Phase 1 - Post-MVP)
4.  **Sécurité de l'API :** Implémentation d'une authentification (JWT/Clé d'API).
5.  **Couverture de Tests :** Atteindre une couverture de tests >80% pour les modules critiques.
6.  **Robustesse :** Ajout de mécanismes de fallback et de "circuit breakers".

## 4. Stack Technique et Contraintes Matérielles
- **Langage Principal :** Python 3.11+.
- **Stack IA :** `insanely-fast-whisper`, `llama-cpp-python`, `piper.exe`.
- **Contraintes :**
    - **100% Offline :** Aucun appel réseau autorisé pour les fonctionnalités de base.
    - **GPU NVIDIA Requis :** Le projet est optimisé pour CUDA. Une configuration dual-GPU (ex: RTX 3090 pour LLM, RTX 4060 Ti pour STT) est idéale.
    - **Configuration Matérielle :** 32Go+ de RAM et 50Go+ de stockage sont recommandés.

## 5. Critères de Succès
- **Performance (Cible pour la fin de la Phase 0) :** Latence totale du pipeline voix-à-voix inférieure à 1.2 secondes.
- **Qualité :** Précision de transcription élevée et voix de synthèse naturelle et claire.
- **Stabilité :** L'application doit fonctionner de manière continue sans crashs ni fuites de mémoire.
- **Sécurité :** L'API future doit être sécurisée avant toute exposition, même locale.