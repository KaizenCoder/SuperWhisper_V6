# 📋 PLAN DE DÉVELOPPEMENT - PIPELINE COMPLET SUPERWHISPER V6

**Version :** 5.2 PIPELINE VOIX-À-VOIX COMPLET OPTIMISÉ  
**Date :** 13 juin 2025  
**Configuration :** RTX 3090 Unique (24GB VRAM)  
**Durée prévue :** 2 jours - VALIDATION HUMAINE OBLIGATOIRE  
**Objectif :** Pipeline complet STT→LLM→TTS avec validation utilisateur  
**Code :** Version 1.1 corrigée OBLIGATOIRE  
**Optimisations :** Ordonnancement amélioré + tâches critiques ajoutées  

---

## 🎯 **OBJECTIFS PIPELINE COMPLET**

### **Vision Produit**
Créer un **pipeline voix-à-voix complet** intégrant les composants STT et TTS validés avec un modèle de langage local pour une expérience conversationnelle fluide < 1.2s end-to-end.

### **Objectifs Techniques Spécifiques**
- **O1** : Latence end-to-end < 1.2s pour 95ᵉ percentile glissant sur les 100 derniers tours
- **O2** : Pipeline STT→LLM→TTS fonctionnel avec fallbacks gracieux
- **O3** : Intégration UnifiedTTSManager asynchrone sans régression
- **O4** : Validation humaine pipeline complet en conditions réelles
- **O5** : Métriques Prometheus temps réel avec dashboard Grafana

---

## 📅 **PLANNING OPTIMISÉ - 2 JOURS**

### **🚀 JOUR 1 - INFRASTRUCTURE & ORCHESTRATEUR (9h + 1h tampon)**

#### **PRÉ-FLIGHT CHECKS (30 min) - 08:00-08:30**
- **assert_gpu_env** (5 min) : Validation CUDA_VISIBLE_DEVICES='1' obligatoire
- **validate_audio_devices** (15 min) : Permissions Windows + énumération devices
- **start_llm** (10 min) : Démarrage serveur LLM avec health-check complet

#### **ÉTAPE 1.1 - Structure Projet (30 min) - 08:30-09:00**
**Créer :** Structure répertoire `PIPELINE/` avec sous-modules
```
PIPELINE/
├── pipeline_orchestrator.py    # Orchestrateur principal
├── config/
│   ├── pipeline.yaml          # Configuration complète
│   └── llm_models.yaml        # Configuration modèles LLM
├── utils/
│   ├── audio_converter.py     # Conversion audio bytes→np.ndarray
│   └── metrics_collector.py   # Collecte métriques Prometheus
└── tests/
    ├── test_pipeline_unit.py  # Tests unitaires
    └── test_pipeline_integration.py # Tests intégration
```

#### **ÉTAPE 1.2 - Configuration LLM (45 min) - 09:00-09:45**
**Créer :** Configuration serveur LLM local avec health-check robuste
- **LLM endpoint** : Configuration vLLM/llama.cpp local
- **Health-check** : `await until /health 200 OK AND first /completions < 5s`
- **quantize_llm** : Génération version Q4_K_M si VRAM sous tension
- **Timeout handling** : Gestion gracieuse des timeouts LLM

#### **ÉTAPE 1.3 - TTS Async Adapter (1h) - 09:45-10:45**
**Créer :** Adaptateur sync→async pour UnifiedTTSManager Phase 3
- **Shim adaptateur** : Wrapper async sans modification backends
- **Interface async** : `async def synthesize_async(text: str) → TTSResult`
- **Conversion audio** : `_wav_bytes_to_numpy()` pour bytes → np.ndarray
- **Tests unitaires** : Validation adaptateur avec backends existants

#### **ÉTAPE 1.4 - PipelineOrchestrator Principal (2h) - 10:45-12:45**
**🚨 OBLIGATOIRE :** Utiliser le code v1.1 corrigé du prompt  
**Créer :** `PIPELINE/pipeline_orchestrator.py` (code exhaustif fourni dans prompt)

**CORRECTIONS v1.1 APPLIQUÉES :**
- ✅ Import TTS : `TTSManager` → `UnifiedTTSManager`
- ✅ Conversion audio : `_wav_bytes_to_numpy()` pour bytes → np.ndarray
- ✅ Gestion erreurs TTS : Vérification `tts_result.success`
- ✅ Import STT optimisé : `OptimizedUnifiedSTTManager`
- ✅ Configuration YAML : Support complet

#### **PAUSE DÉJEUNER (1h) - 12:45-13:45**

#### **ÉTAPE 1.5 - Tests Unitaires Orchestrateur (1h) - 13:45-14:45**
**Créer :** Suite tests unitaires pipeline orchestrateur
- **Tests composants** : STT, LLM, TTS isolément
- **Tests workers** : Validation workers asynchrones
- **Tests fallbacks** : Gestion erreurs et fallbacks
- **min_coverage** : 90% minimum requis

#### **ÉTAPE 1.6 - Métriques & Monitoring (1h) - 14:45-15:45**
**Créer :** Infrastructure métriques Prometheus + Grafana
- **push_metrics_grafana** : Premier dashboard pipeline
- **Métriques temps réel** : Latence, throughput, erreurs
- **Collecteur** : `metrics_collector.py` avec export Prometheus
- **Dashboard** : Visualisation temps réel performance pipeline

#### **ÉTAPE 1.7 - Scripts Utilitaires (45 min) - 15:45-16:30**
**Créer :** Scripts support et démonstration
- **demo_pipeline.py** : Démonstration pipeline complet
- **benchmark_pipeline.py** : Tests performance automatisés
- **validate_setup.py** : Validation configuration complète

#### **TAMPON JOUR 1 (1h30) - 16:30-18:00**
**Réservé :** Débogage imprévu, ajustements, finalisation

---

### **🧪 JOUR 2 - TESTS & VALIDATION (8h)**

#### **ÉTAPE 2.1 - Tests Intégration (2h) - 08:00-10:00**
**Créer :** Tests intégration pipeline complet
- **Tests end-to-end** : STT→LLM→TTS complet
- **Tests performance** : Validation latence < 1.2s
- **Tests robustesse** : Gestion erreurs, fallbacks
- **Tests conditions réelles** : Microphone, environnement normal

#### **ÉTAPE 2.2 - Optimisation Performance (1h30) - 10:00-11:30**
**Optimiser :** Performance pipeline selon métriques
- **Script optimisation** : `python scripts/optimise.py --latency 1200`
- **Profiling** : Identification goulots d'étranglement
- **Tuning** : Ajustement paramètres pour performance optimale
- **Validation** : Confirmation objectifs performance atteints

#### **ÉTAPE 2.3 - Validation Humaine Pipeline (2h) - 11:30-13:30**
**Valider :** Pipeline complet avec utilisateur réel
- **Tests conversationnels** : Dialogue voix-à-voix complet
- **Validation précision** : STT + LLM + TTS ensemble
- **Tests conditions variables** : Bruit, distance, accents
- **Documentation résultats** : Rapport validation détaillé

#### **PAUSE DÉJEUNER (1h) - 13:30-14:30**

#### **ÉTAPE 2.4 - Security Review (1h) - 14:30-15:30**
**Valider :** Sécurité et conformité SI interne
- **Audit dépendances** : Validation aucune DLL/EXE inconnue
- **Review sécurité** : Validation conformité standards
- **Documentation sécurité** : Rapport conformité
- **Approbation** : Validation PM/Tech-Writer

#### **ÉTAPE 2.5 - Documentation Finale (1h30) - 15:30-17:00**
**Finaliser :** Documentation complète projet
- **Guide utilisateur** : Manuel utilisation pipeline
- **Documentation technique** : Architecture et APIs
- **Guide déploiement** : Instructions installation/configuration
- **Changelog** : Historique modifications et améliorations

---

## 🎯 **LIVRABLES JOUR 1**
- `PIPELINE/pipeline_orchestrator.py` implémenté avec **code v1.1 corrigé OBLIGATOIRE**
- Interface UnifiedTTSManager Phase 3 adaptée pipeline (correction import)
- Conversion audio bytes→np.ndarray implémentée
- Serveur LLM configuré avec health-check robuste
- Tests unitaires infrastructure > 90% coverage
- Scripts installation et configuration opérationnels
- Métriques Prometheus + dashboard Grafana initial

## 🎯 **LIVRABLES JOUR 2**
- Tests intégration pipeline complet > 90% coverage
- Validation humaine pipeline voix-à-voix réussie
- Performance < 1.2s end-to-end validée
- Security review approuvée
- Documentation complète finalisée
- Pipeline production-ready livré

---

## 🔧 **CONFIGURATION TECHNIQUE DÉTAILLÉE**

### **Configuration LLM Serveur**
```yaml
llm:
  endpoint: "http://localhost:8000"
  model: "llama-3-8b-instruct"
  timeout: 30.0
  health_check:
    endpoint: "/health"
    timeout: 5.0
    retry_count: 3
  quantization:
    enabled: true
    format: "Q4_K_M"
    auto_quantize: true
```

### **Configuration Pipeline**
```yaml
pipeline:
  max_queue_size: 10
  worker_timeout: 30.0
  enable_metrics: true
  metrics_port: 9091
  fallback_enabled: true
  audio:
    sample_rate: 22050
    format: "wav"
    conversion: "auto"
```

### **Pré-flight Checks**
```bash
# assert_gpu_env
python - <<'PY'
import os, sys
assert os.getenv("CUDA_VISIBLE_DEVICES")=="1", "Bad GPU mapping"
PY

# validate_audio_devices
python scripts/validate_audio_devices.py --check-permissions --enumerate-devices

# start_llm avec health-check
python scripts/start_llm.py --wait-healthy --timeout 60
```

---

## 📊 **MÉTRIQUES ET KPI**

### **Objectifs Performance**
- **Latence end-to-end** : < 1.2s pour 95ᵉ percentile glissant sur 100 derniers tours
- **Disponibilité** : > 99% uptime pipeline
- **Précision STT** : > 95% accuracy conditions normales
- **Qualité TTS** : > 4.0/5.0 évaluation humaine
- **Throughput** : > 10 tours/minute soutenus

### **Métriques Monitoring**
- **pipeline_latency_seconds** : Latence end-to-end par tour
- **pipeline_requests_total** : Nombre total requêtes pipeline
- **pipeline_errors_total** : Nombre erreurs par composant
- **stt_processing_time** : Temps traitement STT
- **llm_processing_time** : Temps traitement LLM
- **tts_processing_time** : Temps traitement TTS

---

## 🚨 **POINTS CRITIQUES ET RISQUES**

### **Risques Techniques Identifiés**
1. **LLM Download** : Modèle 8B Q4 ≈ 4Go peut bloquer démarrage
2. **TTS Async** : Refactorisation sync→async peut être complexe
3. **Audio Conversion** : Conversion bytes→np.ndarray peut introduire latence
4. **GPU Memory** : VRAM 3090 peut être sous tension avec LLM+STT+TTS

### **Mitigations Appliquées**
1. **Health-check LLM** : Attente confirmation modèle prêt avant pipeline
2. **Shim Adapter** : Wrapper async sans modification backends existants
3. **Conversion Optimisée** : `_wav_bytes_to_numpy()` optimisée performance
4. **Quantization** : Q4_K_M automatique si VRAM insuffisante

---

## ✅ **CRITÈRES DE SUCCÈS**

### **Validation Technique**
- [ ] Pipeline STT→LLM→TTS fonctionnel end-to-end
- [ ] Latence < 1.2s validée sur 100 tours consécutifs
- [ ] Tests unitaires > 90% coverage
- [ ] Tests intégration > 90% coverage
- [ ] Métriques Prometheus opérationnelles

### **Validation Humaine**
- [ ] Dialogue voix-à-voix fluide validé utilisateur
- [ ] Précision transcription > 95% conditions normales
- [ ] Qualité synthèse vocale > 4.0/5.0 évaluation
- [ ] Robustesse validée conditions variables

### **Validation Opérationnelle**
- [ ] Security review approuvée
- [ ] Documentation complète finalisée
- [ ] Scripts déploiement opérationnels
- [ ] Pipeline production-ready livré

---

## 🎯 **GO/NO-GO DÉCISION**

### **Critères Go**
- Tests techniques > 90% réussis
- Validation humaine positive
- Performance objectifs atteints
- Security review approuvée
- Documentation complète

### **Critères No-Go**
- Latence > 1.5s persistante
- Échec validation humaine
- Problèmes sécurité bloquants
- Tests < 80% coverage

### **Meeting Go/No-Go**
- **Date** : Fin Jour 2 (17:00)
- **Participants** : PM, Tech Lead, QA, Utilisateur
- **meeting_notes** : `docs/go_no_go_20250615.md`
- **Décision** : Go production ou itération supplémentaire

---

*Plan de développement SuperWhisper V6 - Pipeline Complet*  
*Version 5.2 Optimisée - 13 juin 2025* 