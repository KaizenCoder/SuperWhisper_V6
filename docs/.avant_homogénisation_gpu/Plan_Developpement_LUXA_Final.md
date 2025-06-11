# Plan de Développement Final - LUXA

**Version :** 1.0  
**Date :** 11 juin 2025  
**Objectif :** Roadmap de développement séquentiel pour finaliser LUXA en version production

---

## Vue d'Ensemble Stratégique

### Philosophie de Développement
- **Validation Continue :** Chaque Manager est testé en conditions réelles avant passage au suivant
- **Préservation des Acquis :** Architecture sécurité/monitoring/robustesse maintenue
- **Approche Incrémentale :** Implémentation séquentielle pour minimiser les risques

### Architecture Cible Confirmée
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ RobustSTTManager│───▶│EnhancedLLMManager│───▶│UnifiedTTSManager│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  VAD Optimisé   │    │ Circuit Breakers│    │Multi-Backends   │
│  + Fallbacks    │    │ + Contexte      │    │ + Métriques     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Phase 1 : Corrections Critiques et Stabilisation
**Priorité :** ABSOLUE  
**Durée Estimée :** 10 jours  
**Objectif :** Rendre le build fonctionnel et remplacer les composants MVP par leurs versions robustes

### ✅ Tâche 1 : Correction Import Bloquant (TERMINÉE)
- **Statut :** ✅ COMPLÉTÉ le 11 juin 2025
- **Action :** Suppression import inutile `require_api_key` dans `master_handler_robust.py`
- **Résultat :** 115 tests débloqués, projet fonctionnel

### 🎯 Tâche 2 : Implémentation et Validation RobustSTTManager
**Priorité :** CRITIQUE IMMÉDIATE  
**Durée :** 3 jours  

#### **Sous-tâche 2.1 : Implémentation du Manager**
- **Fichier à créer :** `STT/stt_manager_robust.py`
- **Fonctionnalités Clés :**
  ```python
  class RobustSTTManager:
      - Sélection GPU automatique optimale
      - Chaîne de fallback multi-modèles
      - Gestion VRAM intelligente avec clear_cache
      - Métriques temps réel (latence, erreurs, succès)
      - Conversion audio robuste (bytes ↔ numpy)
      - Intégration VAD existant
  ```

#### **Sous-tâche 2.2 : Adaptation Script Test**
- **Fichier à modifier :** `tests/test_realtime_audio_pipeline.py`
- **Validation Obligatoire :**
  - Test avec microphone physique réel
  - Phrase de validation : "Ceci est un test de validation du nouveau gestionnaire robuste"
  - Assertions sémantiques sur transcription
  - Métriques de performance enregistrées

#### **Sous-tâche 2.3 : Intégration Orchestrateur**
- **Fichier à modifier :** `run_assistant.py`
- **Changements :**
  ```python
  from STT.stt_manager_robust import RobustSTTManager
  stt_handler = RobustSTTManager(config['stt'])
  await stt_handler.initialize()
  ```

#### **Critères d'Acceptation Tâche 2 :**
- ✅ `pytest -v -s tests/test_realtime_audio_pipeline.py` passe avec succès
- ✅ Transcription micro réel fonctionnelle avec assertions validées
- ✅ Métriques latency < 300ms pour audio court
- ✅ Fallback GPU→CPU opérationnel
- ✅ Ancien `stt_handler.py` supprimé

### 🎯 Tâche 3 : Implémentation et Validation UnifiedTTSManager
**Priorité :** HAUTE  
**Durée :** 3 jours

#### **Sous-tâche 3.1 : Consolidation Handlers TTS**
- **Problème Actuel :** 15+ handlers TTS fragmentés
- **Solution :** Manager unifié multi-backends
- **Fichier à créer :** `TTS/tts_manager_unified.py`
- **Architecture :**
  ```python
  class UnifiedTTSManager:
      backends = {
          'piper': PiperTTSHandler,
          'sapi': SAPIHandler, 
          'espeak': ESpeakHandler
      }
      - Auto-sélection backend optimal
      - Fallback chain configurée
      - Quality scoring et adaptation
      - Cache synthèse intelligente
  ```

#### **Sous-tâche 3.2 : Tests et Validation**
- **Nouveaux tests :** `tests/test_unified_tts_manager.py`
- **Validation :** Synthèse multi-backends + qualité audio

#### **Critères d'Acceptation Tâche 3 :**
- ✅ Synthèse audio fonctionnelle tous backends
- ✅ Latence < 200ms pour phrase courte
- ✅ Tests automatisés passent
- ✅ Anciens handlers TTS supprimés

### 🎯 Tâche 4 : Implémentation et Validation EnhancedLLMManager
**Priorité :** HAUTE  
**Durée :** 4 jours

#### **Sous-tâche 4.1 : Manager LLM Avancé**
- **Problème Actuel :** Handler LLM basique sans contexte
- **Solution :** Manager avec historique conversationnel
- **Fichier à créer :** `LLM/llm_manager_enhanced.py`
- **Fonctionnalités :**
  ```python
  class EnhancedLLMManager:
      - Gestion contexte conversationnel
      - Hot-swapping modèles (VRAM optimization)
      - Circuit breakers intégrés
      - Streaming responses
      - Métriques tokens/s et VRAM usage
  ```

#### **Sous-tâche 4.2 : Tests Unitaires Contexte**
- **Tests :** Gestion conversation multi-tours
- **Validation :** Cohérence contextuelle + performances

#### **Critères d'Acceptation Tâche 4 :**
- ✅ Conversation contextuelle fonctionnelle
- ✅ Hot-swapping sans crash
- ✅ Latence < 500ms réponse standard
- ✅ Circuit breakers opérationnels

---

## Phase 2 : Extensions Intelligentes sur Base Saine
**Priorité :** MOYENNE  
**Durée Estimée :** 14 jours  
**Objectif :** Implémenter fonctionnalités à forte valeur ajoutée

### 🚀 Tâche 5 : Gestion Multi-Modèles "Hot-Swap"
**Durée :** 5 jours
- **Objectif :** Optimisation dynamique VRAM
- **Implémentation :**
  ```python
  class ModelSwapManager:
      - Détection charge VRAM temps réel
      - Swap automatique Large↔Medium↔Small
      - Prédiction besoins par contexte
      - Persistence état conversation
  ```

### 🚀 Tâche 6 : Interface Sélection Microphone
**Durée :** 4 jours
- **Objectif :** UX amélioration périphériques audio
- **Composants :**
  - Détection automatique périphériques
  - Interface sélection utilisateur
  - Tests qualité micro temps réel
  - Persistance préférences

### 🚀 Tâche 7 : Optimisations GPU SuperWhisper2
**Durée :** 5 jours
- **Objectif :** Performance maximale héritée
- **Techniques :**
  ```python
  - Memory Pinning pour accès GPU rapide
  - CUDA Streams parallelisation
  - Batch processing intelligent
  - Préchargement modèles optimisé
  ```

---

## Phase 3 : Tests de Performance et Finalisation
**Priorité :** FINALE  
**Durée Estimée :** 10 jours  
**Objectif :** Validation production et monitoring complet

### 📊 Tâche 8 : Suite Benchmarks Performance
**Durée :** 4 jours
- **Métriques Cibles :**
  - Latence pipeline < 1.2s (objectif < 1.0s)
  - WER STT < 5% français
  - Qualité TTS MOS > 4.0
  - Utilisation VRAM optimale

### 🧪 Tâche 9 : Tests de Charge et Résistance
**Durée :** 3 jours
- **Scénarios :**
  - Conversation 1h+ continue
  - Pics charge simultanés
  - Récupération après crash
  - Memory leaks détection

### 📈 Tâche 10 : Dashboard Monitoring Grafana
**Durée :** 3 jours
- **Métriques Temps Réel :**
  - Performance pipeline complet
  - Usage GPU/CPU/RAM
  - Qualité transcriptions
  - Erreurs et recovery

---

## Planning et Jalons

### Calendrier Prévisionnel
```
Phase 1 (J+0 → J+10) : Stabilisation CRITIQUE
├─ J+0  : ✅ Import fix terminé
├─ J+3  : 🎯 RobustSTTManager + validation micro
├─ J+6  : 🎯 UnifiedTTSManager intégré
├─ J+10 : 🎯 EnhancedLLMManager + pipeline complet

Phase 2 (J+11 → J+24) : Extensions AVANCÉES  
├─ J+16 : 🚀 Hot-swapping LLM opérationnel
├─ J+20 : 🚀 Interface sélection microphone
├─ J+24 : 🚀 Optimisations GPU intégrées

Phase 3 (J+25 → J+34) : Production FINALE
├─ J+28 : 📊 Benchmarks complets validés
├─ J+31 : 🧪 Tests charge réussis
├─ J+34 : 📈 Dashboard monitoring finalisé
```

### Points de Contrôle Obligatoires
- **Checkpoint Phase 1 (J+10) :** Pipeline voix-à-voix fonctionnel complet
- **Checkpoint Phase 2 (J+24) :** Fonctionnalités avancées intégrées
- **Checkpoint Final (J+34) :** Version production validée

---

## Ressources et Dépendances

### Dépendances Techniques
- **Hardware :** GPU NVIDIA 8GB+ VRAM (idéal dual-GPU)
- **Software :** CUDA 11.8+, Python 3.11+, PyTorch 2.0+
- **Modèles :** Whisper (tiny→base), LLama models, Piper voices

### Ressources Humaines
- **Développeur Principal :** Implémentation Managers
- **Testeur QA :** Validation micro + benchmarks
- **DevOps :** Intégration monitoring + déploiement

### Outils de Suivi
- **Tests Automatisés :** pytest avec coverage >80%
- **CI/CD :** GitHub Actions avec validation continue
- **Monitoring :** Prometheus + Grafana temps réel
- **Documentation :** Mise à jour continue journal développement

---

## Critères de Succès Global

### Objectifs Techniques SMART
1. **Performance :** Pipeline < 1.2s validé par benchmarks
2. **Qualité :** WER < 5%, MOS > 4.0 mesurés en conditions réelles
3. **Robustesse :** 99.5% uptime sur tests 24h
4. **Sécurité :** Architecture JWT/API préservée intégralement
5. **Monitoring :** Dashboard temps réel opérationnel

### Validation Finale
- ✅ Démo complète assistant vocal fonctionnel
- ✅ Suite tests automatisés 100% passants  
- ✅ Documentation technique complète
- ✅ Dashboard monitoring déployé
- ✅ Version production prête déploiement

---

**Équipe Projet LUXA**  
**Dernière Mise à Jour :** 11 juin 2025  
**Prochaine Révision :** Checkpoint Phase 1 (J+10) 