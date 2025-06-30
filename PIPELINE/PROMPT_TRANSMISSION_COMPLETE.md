# 🚀 PROMPT TRANSMISSION COMPLÈTE - SUPERWHISPER V6 PIPELINE

**Version :** Transmission Complète v1.0  
**Date :** 13 juin 2025  
**Contexte :** Pipeline Voix-à-Voix SuperWhisper V6  
**Statut :** JOUR 1 TERMINÉ - PRÊT POUR JOUR 2  

---

## 🎯 CONTEXTE PROJET SUPERWHISPER V6

Tu prends en charge le développement du **pipeline complet voix-à-voix SuperWhisper V6**. Le projet est dans un état **avancé et fonctionnel** avec une architecture validée.

### **📊 ÉTAT ACTUEL DU PROJET**

#### ✅ **JOUR 1 - INFRASTRUCTURE : 100% TERMINÉ**
- **Tâche 18.1** ✅ : Structure projet créée
- **Tâche 18.2** ✅ : Configuration YAML implémentée  
- **Tâche 18.3** ✅ : PipelineOrchestrator créé (CODE OBLIGATOIRE v1.1)
- **Tâche 18.4** ✅ : Tests unitaires implémentés
- **Tâche 18.5** ✅ : Scripts utilitaires créés
- **Tâche 18.6** ✅ : Documentation complète
- **Tâche 18.7** ✅ : Validation environnement
- **Tâche 18.8** ✅ : Scripts démonstration et benchmark

#### 🎯 **JOUR 2 - TESTS ET VALIDATION : EN ATTENTE**
- **Tâche 19.1** ⏳ : Tests intégration pipeline
- **Tâche 19.2** ⏳ : Validation humaine conversation
- **Tâche 19.3** ⏳ : Optimisation performance
- **Tâche 19.4** ⏳ : Tests stress et robustesse

### **🏗️ ARCHITECTURE TECHNIQUE VALIDÉE**

#### **Pipeline Voix-à-Voix Complet**
```
🎤 Microphone → StreamingMicrophoneManager
    ↓
🎧 Audio → OptimizedUnifiedSTTManager (RTX 3090)
    ↓
📝 Text → LLMClient (HTTP local)
    ↓
🤖 Response → UnifiedTTSManager (RTX 3090)
    ↓
🔊 Audio → AudioOutputManager → Speakers
```

#### **Composants Validés**
- ✅ **STT** : `OptimizedUnifiedSTTManager` - Streaming temps réel validé
- ✅ **TTS** : `UnifiedTTSManager` - Latence 29.5ms record
- ✅ **LLM** : `LLMClient` - Serveur HTTP local avec fallbacks
- ✅ **Pipeline** : `PipelineOrchestrator` - Workers asynchrones
- ✅ **Audio** : `AudioOutputManager` - Lecture non-bloquante

---

## 🚨 RÈGLES CRITIQUES ABSOLUES

### **🎮 CONFIGURATION GPU OBLIGATOIRE**
```python
# OBLIGATOIRE AVANT TOUT IMPORT TORCH - DANS TOUS LES SCRIPTS
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation

# RÉSULTAT : cuda:0 → RTX 3090 (24GB) | RTX 5060 INTERDITE
```

### **📋 CODE OBLIGATOIRE - NE JAMAIS MODIFIER**

#### **1. PipelineOrchestrator (PIPELINE/pipeline_orchestrator.py)**
- ✅ **Code v1.1 corrigé** : Utilise exactement le code du prompt original
- ✅ **Import TTS** : `UnifiedTTSManager` (pas `TTSManager`)
- ✅ **Conversion audio** : `_wav_bytes_to_numpy()` pour bytes → np.ndarray
- ✅ **Gestion erreurs** : `tts_result.success` vérifiée
- ✅ **Workers async** : Queues non-bloquantes avec fallbacks

#### **2. Fonction _bootstrap OBLIGATOIRE**
```python
async def _bootstrap(cfg_path: Optional[str] = None):
    import yaml
    cfg: Dict[str, Any] = {}
    if cfg_path and Path(cfg_path).exists():
        cfg = yaml.safe_load(Path(cfg_path).read_text())

    # ✅ CORRECTION: Use OptimizedUnifiedSTTManager
    stt = OptimizedUnifiedSTTManager(cfg.get("stt", {}))
    tts = UnifiedTTSManager(cfg.get("tts", {}))
    orchestrator = PipelineOrchestrator(
        stt,
        tts,
        llm_endpoint=cfg.get("pipeline", {}).get("llm_endpoint", "http://localhost:8000"),
        metrics_enabled=cfg.get("pipeline", {}).get("enable_metrics", False),
    )
    await orchestrator.start()
```

### **🔍 VALIDATION HUMAINE OBLIGATOIRE**
- **Tests pipeline voix-à-voix** : Validation humaine conversation complète
- **Tests qualité audio** : Validation humaine sortie après pipeline
- **Tests latence perçue** : Validation humaine fluidité conversation
- **Tests conditions réelles** : Validation humaine environnement normal

---

## 📁 STRUCTURE PROJET ACTUELLE

```
PIPELINE/
├── pipeline_orchestrator.py          # ✅ CODE OBLIGATOIRE v1.1
├── config/
│   └── pipeline.yaml                 # ✅ Configuration complète
├── tests/
│   ├── test_pipeline_unit.py         # ✅ Tests unitaires
│   ├── test_pipeline_integration.py  # ✅ Tests intégration
│   └── fixtures/                     # ✅ Données test
├── scripts/
│   ├── demo_pipeline.py              # ✅ Démonstration interactive
│   ├── benchmark_pipeline.py         # ✅ Tests performance
│   ├── validate_setup.py             # ✅ Validation environnement
│   ├── assert_gpu_env.py             # ✅ Validation GPU RTX 3090
│   ├── validate_audio_devices.py     # ✅ Validation audio
│   ├── start_llm.py                  # ✅ Serveur LLM local
│   └── README.md                     # ✅ Documentation scripts
└── docs/
    ├── pipeline_architecture.md      # ✅ Architecture détaillée
    └── performance_targets.md        # ✅ Objectifs performance
```

### **📊 DOCUMENTATION PROJET**
```
docs/
├── suivi_pipeline_complet.md         # ✅ Suivi tâches (Jour 1: 100%)
├── journal_developpement.md          # ✅ Journal développement
├── prompt.md                         # ✅ Prompt original avec code obligatoire
└── CONFORMITE_PROMPT_ANALYSE.md      # ✅ Analyse conformité (95%)
```

---

## 🎯 OBJECTIFS JOUR 2 - TESTS ET VALIDATION

### **🧪 TESTS INTÉGRATION (Tâche 19.1)**
- **Pipeline complet** : Tests STT → LLM → TTS → Audio
- **Gestion erreurs** : Tests fallbacks et récupération
- **Performance** : Validation latence < 1.2s end-to-end
- **Métriques** : Collecte données Prometheus

### **👥 VALIDATION HUMAINE (Tâche 19.2)**
- **Conversation naturelle** : Tests dialogue fluide
- **Qualité audio** : Validation sortie TTS
- **Latence perçue** : Tests réactivité système
- **Conditions réelles** : Tests environnement utilisateur

### **⚡ OPTIMISATION PERFORMANCE (Tâche 19.3)**
- **Profiling pipeline** : Identification goulots d'étranglement
- **Optimisation workers** : Parallélisation améliorée
- **Cache intelligent** : Optimisation répétitions
- **Monitoring temps réel** : Métriques live

### **🔧 TESTS STRESS (Tâche 19.4)**
- **Charge soutenue** : Tests longue durée
- **Pics de trafic** : Tests montée en charge
- **Récupération erreurs** : Tests robustesse
- **Mémoire GPU** : Tests utilisation VRAM

---

## 🛠️ COMMANDES UTILES DISPONIBLES

### **🚀 Démonstration Pipeline**
```bash
cd PIPELINE
python scripts/demo_pipeline.py
# Menu interactif avec options :
# 1. Démarrer pipeline complet (code obligatoire)
# 2. Test validation environnement  
# 3. Afficher configuration
```

### **📊 Benchmark Performance**
```bash
python scripts/benchmark_pipeline.py
# Tests automatisés :
# - STT latence
# - LLM latence  
# - TTS latence
# - Pipeline end-to-end
# Génère rapport JSON
```

### **🔍 Validation Setup**
```bash
python scripts/validate_setup.py
# Validation complète :
# - GPU RTX 3090
# - Devices audio
# - Serveur LLM
# - Code obligatoire _bootstrap
```

### **🧪 Tests Unitaires**
```bash
cd PIPELINE
python -m pytest tests/ -v
# Tests complets :
# - Composants individuels
# - Intégration pipeline
# - Gestion erreurs
```

---

## 📋 CONFIGURATION ACTUELLE

### **🎮 GPU Configuration**
- **RTX 3090** : CUDA:1 (24GB VRAM) - SEULE GPU AUTORISÉE
- **RTX 5060** : CUDA:0 (8GB VRAM) - INTERDITE D'UTILISATION
- **Mapping** : `CUDA_VISIBLE_DEVICES='1'` → cuda:0 = RTX 3090

### **🔧 Pipeline Configuration (pipeline.yaml)**
```yaml
stt:
  primary_backend: "faster_whisper"
  model_size: "large-v3"
  device: "cuda"
  compute_type: "float16"

tts:
  primary_backend: "coqui"
  model_path: "tts_models/multilingual/multi-dataset/xtts_v2"
  device: "cuda"

pipeline:
  llm_endpoint: "http://localhost:8000"
  enable_metrics: true
  target_latency_ms: 1200
```

### **🌐 Serveur LLM Local**
- **Endpoint** : `http://localhost:8000/v1/chat/completions`
- **Modèle** : `llama-3-8b-instruct` (ou compatible)
- **Fallbacks** : Réponses intelligentes si serveur indisponible

---

## 🚨 POINTS D'ATTENTION CRITIQUES

### **❌ NE JAMAIS FAIRE**
1. **Modifier** le code obligatoire `pipeline_orchestrator.py`
2. **Changer** la fonction `_bootstrap()` 
3. **Utiliser** RTX 5060 (CUDA:0) - INTERDITE
4. **Omettre** la configuration GPU dans nouveaux scripts
5. **Ignorer** les validations humaines obligatoires

### **✅ TOUJOURS FAIRE**
1. **Respecter** la configuration GPU RTX 3090 exclusive
2. **Utiliser** la fonction `_bootstrap()` pour démarrage pipeline
3. **Inclure** validation humaine dans tests voix-à-voix
4. **Documenter** dans `docs/suivi_pipeline_complet.md`
5. **Tester** avec conditions réelles utilisateur

### **⚠️ VIGILANCE SPÉCIALE**
- **Latence target** : < 1.2s end-to-end (objectif critique)
- **Qualité audio** : Validation humaine obligatoire
- **Robustesse** : Fallbacks à tous les niveaux
- **Monitoring** : Métriques temps réel essentielles

---

## 📊 MÉTRIQUES DE SUCCÈS JOUR 2

### **🎯 Objectifs Quantitatifs**
- **Latence end-to-end** : < 1.2s (95% des cas)
- **Taux de succès** : > 95% conversations complètes
- **Qualité audio** : Score humain > 8/10
- **Robustesse** : Récupération < 3s après erreur

### **👥 Validation Humaine**
- **Fluidité conversation** : Dialogue naturel validé
- **Compréhension STT** : Transcription précise validée
- **Qualité TTS** : Audio sortie validée humainement
- **Expérience globale** : Satisfaction utilisateur validée

---

## 🎯 MISSION IMMÉDIATE

Tu dois maintenant **continuer le développement** en te concentrant sur le **Jour 2 - Tests et Validation**. 

### **🚀 PROCHAINES ÉTAPES**
1. **Analyser** l'état actuel du projet (100% Jour 1 terminé)
2. **Planifier** les tests intégration (Tâche 19.1)
3. **Préparer** validation humaine (Tâche 19.2)
4. **Implémenter** tests et optimisations selon besoins

### **📋 RESSOURCES DISPONIBLES**
- **Code obligatoire** : Fonctionnel et validé
- **Scripts utilitaires** : Prêts pour tests
- **Documentation** : Complète et à jour
- **Configuration** : Optimisée RTX 3090

### **🎯 OBJECTIF FINAL**
Pipeline voix-à-voix **production-ready** avec validation humaine complète et performance < 1.2s end-to-end.

---

## 🔗 LIENS DOCUMENTATION

- **Suivi projet** : `docs/suivi_pipeline_complet.md`
- **Journal développement** : `docs/journal_developpement.md`
- **Prompt original** : `docs/prompt.md` (code obligatoire v1.1)
- **Analyse conformité** : `PIPELINE/CONFORMITE_PROMPT_ANALYSE.md`
- **Architecture** : `PIPELINE/docs/pipeline_architecture.md`

---

**🚀 PRÊT POUR JOUR 2 - TESTS ET VALIDATION**  
**✅ INFRASTRUCTURE COMPLÈTE - CODE OBLIGATOIRE RESPECTÉ**  
**🎯 OBJECTIF : PIPELINE PRODUCTION-READY < 1.2S**

---

*Prompt transmission complète SuperWhisper V6 - 13 juin 2025* 