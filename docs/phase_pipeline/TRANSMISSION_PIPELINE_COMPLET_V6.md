# 🚀 **PROMPT DE TRANSMISSION - PIPELINE COMPLET SUPERWHISPER V6**

**Date de transmission :** 13 juin 2025  
**Version projet :** 6.0.0-beta  
**Statut :** ✅ **STT VALIDÉ** - 🎯 **PIPELINE COMPLET PRÊT À IMPLÉMENTER**  
**Configuration :** RTX 3090 Unique (24GB VRAM) - CUDA_VISIBLE_DEVICES='1'  

---

## 🎯 **MISSION IMMÉDIATE**

Vous devez **implémenter le pipeline complet voix-à-voix** SuperWhisper V6 en utilisant la **version 5.2 OPTIMISÉE** avec code v1.1 corrigé obligatoire.

### **✅ État Actuel Validé**
- **Phase 4 STT** : ✅ **VALIDÉ UTILISATEUR** avec streaming temps réel opérationnel
- **Phase 3 TTS** : ✅ **VALIDÉ** avec performance record 29.5ms latence
- **Architecture** : StreamingMicrophoneManager + UnifiedSTTManager + TTSManager opérationnels
- **Tests** : STT 6/6 réussis, TTS 88.9% succès, RTX 3090 configurée

### **❌ Pipeline Complet Manquant**
- **STT→LLM→TTS** : Intégration bout-en-bout NON TESTÉE
- **Audio Output** : Lecture finale manquante
- **Performance E2E** : Latence totale <1.2s NON VALIDÉE

---

## 📋 **DOCUMENTS CRITIQUES À CONSULTER**

### **🔴 PRIORITÉ ABSOLUE (À lire en PREMIER)**
1. **`docs/prompt_pipeline_complet.md`** - **PROMPT COMPLET v5.2** avec code v1.1 corrigé OBLIGATOIRE
2. **`docs/dev_plan_pipeline_complet.md`** - **PLAN OPTIMISÉ v5.2** (2 jours + pré-flight checks)
3. **`PIPELINE/pipeline_orchestrator.py`** - **CODE v1.1 CORRIGÉ** à utiliser OBLIGATOIREMENT
4. **`PIPELINE/config/taskmaster_pipeline.yaml`** - **CONFIG TASKMASTER OPTIMISÉE**

### **🟠 PRIORITÉ HAUTE (Contexte)**
5. **`docs/prd_pipeline_complet.md`** - PRD v5.2 avec exigences optimisées
6. **`docs/suivi_pipeline_complet.md`** - Suivi v5.2 avec planning 2 jours
7. **`docs/support_expert_architecture_complete.md`** - Architecture complète existante
8. **`docs/ON_BOARDING_ia.md`** - Contexte projet et état STT/TTS validés

---

## 🚨 **RÈGLES ABSOLUES OBLIGATOIRES**

### **1. Code Pipeline Orchestrator v1.1 OBLIGATOIRE**
- **UTILISER EXCLUSIVEMENT** le code dans `PIPELINE/pipeline_orchestrator.py`
- **CORRECTIONS v1.1 APPLIQUÉES** : Import TTS, conversion audio, gestion erreurs
- **INTERDICTION** de modifier le code sans validation préalable

### **2. Configuration RTX 3090 CRITIQUE**
```python
# 🚨 CONFIGURATION OBLIGATOIRE - JAMAIS MODIFIER
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIF
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable obligatoire
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
```

### **3. Planning Optimisé 2 Jours STRICT**
- **Jour 1** : Infrastructure + Orchestrateur (9h + 1h30 tampon)
- **Jour 2** : Tests + Validation humaine (8h)
- **Pré-flight checks** : `assert_gpu_env`, `validate_audio_devices`, `start_llm`

---

## 🎯 **OBJECTIFS PERFORMANCE**

### **Cibles Obligatoires**
- **Latence E2E** : < 1.2s pour 95ᵉ percentile glissant sur 100 derniers tours
- **Tests coverage** : > 90% unitaires + intégration
- **Validation humaine** : Conversation voix-à-voix complète OBLIGATOIRE
- **Disponibilité** : > 99% uptime pipeline

### **Architecture Pipeline**
```
Microphone → StreamingMicrophoneManager → UnifiedSTTManager → LLM Local → TTSManager → Audio Output
     ↓              ↓                        ↓                ↓           ↓            ↓
   VAD WebRTC   faster-whisper RTX3090   HTTP Client    Edge/Piper    sounddevice   < 1.2s
```

---

## 🔧 **COMPOSANTS EXISTANTS VALIDÉS**

### **STT (Phase 4 - VALIDÉ)**
- **StreamingMicrophoneManager** : VAD WebRTC streaming temps réel
- **UnifiedSTTManager** : 4 backends avec RTX 3090 optimisé
- **Performance** : RTF 0.159-0.420, latence 853-945ms, 100% couverture

### **TTS (Phase 3 - VALIDÉ)**
- **TTSManager** : 4 backends (Edge, Piper, Coqui, System)
- **Performance** : 29.5ms latence record, 93.1% cache hit rate
- **Qualité** : Tests 88.9% succès, stabilité 100%

### **À INTÉGRER (NOUVEAU)**
- **LLM Local** : Serveur HTTP vLLM/llama.cpp
- **Audio Output** : sounddevice/simpleaudio
- **Pipeline Orchestrator** : Code v1.1 corrigé fourni

---

## 📅 **PLANNING DÉTAILLÉ OPTIMISÉ**

### **🚀 JOUR 1 - Infrastructure (9h + 1h30 tampon)**

#### **Pré-flight Checks (30 min) - 08:00-08:30**
- **assert_gpu_env** (5 min) : `python -c "import os; assert os.getenv('CUDA_VISIBLE_DEVICES')=='1'"`
- **validate_audio_devices** (15 min) : Permissions Windows + énumération
- **start_llm** (10 min) : Health-check `await until /health 200 OK`

#### **Développement (8h30)**
- **1.1** Structure PIPELINE/ (30 min) - 08:30-09:00
- **1.2** Config LLM + health-check (45 min) - 09:00-09:45
- **1.3** TTS Async Adapter (1h) - 09:45-10:45
- **1.4** **PipelineOrchestrator v1.1** (2h) - 10:45-12:45 **CODE OBLIGATOIRE**
- **1.5** Tests unitaires (1h) - 13:45-14:45
- **1.6** Métriques Prometheus (1h) - 14:45-15:45
- **1.7** Scripts utilitaires (45 min) - 15:45-16:30
- **TAMPON** Débogage (1h30) - 16:30-18:00

### **🧪 JOUR 2 - Tests & Validation (8h)**
- **2.1** Tests intégration E2E (2h) - 08:00-10:00
- **2.2** Optimisation performance (1h30) - 10:00-11:30
- **2.3** **Validation humaine** (2h) - 11:30-13:30 **OBLIGATOIRE**
- **2.4** Security review (1h) - 14:30-15:30
- **2.5** Documentation finale (1h30) - 15:30-17:00

---

## 🚨 **POINTS CRITIQUES OPTIMISATIONS**

### **Améliorations v5.2 Appliquées**
1. **LLM Health-check** : Évite blocage téléchargement modèle 8B
2. **TTS Async Shim** : Adaptateur sans refactorisation lourde
3. **Audio Validation** : Permissions Windows avant tests
4. **Tampon Jour 1** : 1h30 pour débogage imprévu
5. **Tests Intégration** : Déplacés Jour 2 pour plus d'air

### **Tâches Critiques Ajoutées**
- **quantize_llm** : Q4_K_M si VRAM sous tension
- **push_metrics_grafana** : Dashboard pipeline
- **security_review** : Validation DLL/EXE
- **Script optimisation** : `python scripts/optimise.py --latency 1200`

---

## 📊 **VALIDATION HUMAINE OBLIGATOIRE**

### **Protocole Validation**
1. **Conversation complète** : Session voix-à-voix 15+ minutes
2. **Tests conditions** : Bruit, distance, accents variables
3. **Évaluation qualité** : Audio sortie naturelle et claire
4. **Mesure latence** : Perception utilisateur < 1.5s acceptable
5. **Rapport détaillé** : Documentation résultats obligatoire

### **Critères Succès**
- **Fluidité** : Conversation naturelle sans coupures
- **Qualité audio** : TTS claire et compréhensible
- **Latence perçue** : Réactivité acceptable utilisateur
- **Satisfaction** : Expérience conversationnelle positive

---

## 🎯 **LIVRABLES FINAUX ATTENDUS**

### **Jour 1**
- `PIPELINE/pipeline_orchestrator.py` avec **code v1.1 OBLIGATOIRE**
- Serveur LLM configuré avec health-check robuste
- Tests unitaires > 90% coverage
- Métriques Prometheus + dashboard Grafana

### **Jour 2**
- Tests intégration E2E > 90% coverage
- **Validation humaine conversation complète RÉUSSIE**
- Performance < 1.2s end-to-end validée
- Documentation finale complète
- **Pipeline production-ready livré**

---

## 🚀 **COMMENCER IMMÉDIATEMENT**

1. **Lire** `docs/prompt_pipeline_complet.md` (code exhaustif v1.1)
2. **Consulter** `docs/dev_plan_pipeline_complet.md` (planning optimisé)
3. **Utiliser** `PIPELINE/pipeline_orchestrator.py` (code corrigé OBLIGATOIRE)
4. **Suivre** planning 2 jours avec pré-flight checks
5. **Valider** humainement conversation complète

---

**🎯 MISSION : Pipeline voix-à-voix complet < 1.2s avec validation humaine réussie**  
**⏰ DÉLAI : 2 jours optimisés**  
**🚨 CODE : Version 1.1 corrigée OBLIGATOIRE**  
**✅ SUCCÈS : STT + TTS validés, pipeline complet à finaliser**

*Transmission SuperWhisper V6 - 13 juin 2025*  
*Version 5.2 OPTIMISÉE - Prêt pour implémentation immédiate* 