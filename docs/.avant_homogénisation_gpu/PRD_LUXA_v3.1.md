# Product Requirements Document (PRD) - LUXA
**Version :** 3.1  
**Date :** 11 juin 2025  
**Objectif :** Finaliser un assistant vocal de niveau production en stabilisant et unifiant l'architecture existante.

## 1. Vue d'Ensemble
LUXA est un assistant vocal local dont le développement a atteint un niveau d'architecture avancé. Ce PRD définit les exigences pour finaliser le produit en se basant sur les recommandations du "Peer Review Complet" et les leçons apprises des projets antérieurs.

## 2. Objectifs du Projet
1. **Stabiliser la Base :** Remplacer les composants du MVP par des "Managers" robustes, testés en conditions réelles.
2. **Atteindre la Performance Cible :** Appliquer les optimisations GPU éprouvées pour garantir une latence inférieure à 1.2s.
3. **Préparer à la Production :** Mettre en place la sécurité, les tests et la robustesse nécessaires pour un déploiement fiable.

## 3. Périmètre et Exigences Techniques

### Exigences Fonctionnelles
1. **Sélection du Microphone par l'Utilisateur :** L'application DOIT permettre à l'utilisateur de choisir son périphérique d'enregistrement parmi une liste claire.
2. **Gestion Dynamique des Modèles LLM :** Le système DOIT supporter le "hot-swapping" de modèles LLM pour optimiser l'utilisation de la VRAM.

### Exigences de Qualité et de Robustesse
3. **Validation en Conditions Réelles :** Chaque composant du pipeline audio (VAD, STT) DOIT être validé via des tests utilisant un microphone physique réel. Le succès de ces tests est un critère d'acceptation non négociable.
4. **Handlers Robustes :** Les handlers du MVP DOIVENT être remplacés par les implémentations robustes (`RobustSTTManager`, `EnhancedLLMManager`, `UnifiedTTSManager`) qui incluent la gestion des erreurs, les fallbacks et les métriques.
5. **Préservation des Acquis de Sécurité :** L'architecture de sécurité (JWT/API Keys) DOIT être préservée et intégrée dans tous les nouveaux développements.

## 4. Architecture Technique

### Stack Technique Confirmée
- **Langage Principal :** Python 3.11+
- **STT :** OpenAI Whisper avec chaîne de fallback
- **LLM :** llama-cpp-python avec gestion multi-modèles
- **TTS :** Piper + backends multiples (SAPI, eSpeak)
- **Sécurité :** JWT + API Keys (préservé)
- **Monitoring :** Prometheus + Grafana (préservé)
- **Robustesse :** Circuit Breakers (préservé)

### Contraintes Matérielles
- **GPU NVIDIA Requis :** Optimisation CUDA prioritaire
- **Configuration Recommandée :** Dual-GPU (RTX 3090 LLM + RTX 4060 Ti STT)
- **VRAM Minimale :** 4GB pour STT, 8GB+ pour LLM
- **100% Offline :** Tous les modèles locaux obligatoires

## 5. Critères de Succès

### Critères Techniques
- La suite de tests `tests/test_realtime_audio_pipeline.py` passe avec succès après chaque modification majeure du STT.
- La couverture de tests sur les nouveaux "Managers" est supérieure à 80%.
- La latence totale du pipeline voix-à-voix est inférieure à 1.2 secondes en conditions réelles.

### Critères de Performance
- **Latence STT :** < 300ms pour transcription courte (<10s audio)
- **Latence LLM :** < 500ms pour réponse contextuelle standard
- **Latence TTS :** < 200ms pour synthèse vocale
- **Latence Totale Pipeline :** < 1.2s (objectif), < 1.5s (acceptable)

### Critères de Qualité
- **WER (Word Error Rate) STT :** < 5% en français, conditions studio
- **Compréhension LLM :** > 90% requêtes contextuelles pertinentes
- **Qualité TTS :** Score MOS > 4.0 (Mean Opinion Score)

## 6. Phases de Développement

### Phase 1 : Stabilisation (Priorité Critique)
- Correction erreurs d'importation ✅
- Implémentation RobustSTTManager + validation micro
- Implémentation UnifiedTTSManager + tests
- Implémentation EnhancedLLMManager + contexte

### Phase 2 : Extensions Avancées
- Hot-swapping LLM dynamique
- Interface sélection microphone
- Optimisations GPU SuperWhisper2

### Phase 3 : Production
- Benchmarks performance complets
- Tests de charge et résistance
- Dashboard monitoring Grafana

## 7. Risques et Mitigations

### Risques Techniques Identifiés
1. **Compatibilité GPU** → Tests multi-configurations
2. **Performance VAD** → Intégration optimisée existante
3. **Gestion Mémoire VRAM** → Hot-swapping et fallbacks
4. **Latence Cible** → Optimisations SuperWhisper2 éprouvées

### Mitigations Stratégiques
- **Validation continue** avec micro réel à chaque étape
- **Préservation acquis** (sécurité, monitoring, robustesse)
- **Tests automatisés** obligatoires pour chaque Manager
- **Métriques temps réel** pour détection précoce régressions

## 8. Livrables et Jalons

### Jalons Phase 1
- **J+0 :** RobustSTTManager opérationnel + test micro ✅
- **J+3 :** UnifiedTTSManager intégré + validation
- **J+7 :** EnhancedLLMManager avec contexte + tests
- **J+10 :** Pipeline complet fonctionnel

### Jalons Phase 2
- **J+14 :** Hot-swapping LLM opérationnel
- **J+18 :** Interface sélection microphone
- **J+21 :** Optimisations GPU intégrées

### Jalons Phase 3
- **J+25 :** Suite benchmarks complète
- **J+28 :** Dashboard monitoring finalisé
- **J+30 :** Version production prête

## 9. Acceptance Criteria

### Critères d'Acceptation Non Négociables
1. ✅ **Test micro réel obligatoire** pour chaque composant audio
2. ✅ **Coverage tests > 80%** pour tous les nouveaux Managers
3. ✅ **Latence < 1.2s** mesurée en conditions réelles
4. ✅ **Zéro régression** fonctionnalités existantes
5. ✅ **Architecture sécurité préservée** intégralement

### Validation Continue
- Exécution automatique `test_realtime_audio_pipeline.py`
- Métriques performance temps réel
- Tests non-régression complets
- Validation manuelle utilisateur final

---

**Approuvé par :** Équipe Technique LUXA  
**Prochaine révision :** Fin Phase 1 (J+10) 