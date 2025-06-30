# 🛠️ Scripts Utilitaires Pipeline SuperWhisper V6

**Version** : Pipeline Complet v1.1  
**Code** : Utilisation OBLIGATOIRE du prompt v1.1  
**GPU** : RTX 3090 (CUDA:1) EXCLUSIVEMENT  

---

## 📋 **SCRIPTS DISPONIBLES**

### **🔍 Scripts de Validation**

#### **`assert_gpu_env.py`** - Validation GPU RTX 3090 Obligatoire
```bash
python PIPELINE/scripts/assert_gpu_env.py
```
- **Fonction** : Valide configuration RTX 3090 obligatoire
- **Vérifications** : CUDA:1, VRAM >20GB, nom GPU RTX 3090
- **Utilisation** : Pré-requis avant tout démarrage pipeline

#### **`validate_audio_devices.py`** - Validation Devices Audio
```bash
python PIPELINE/scripts/validate_audio_devices.py
```
- **Fonction** : Énumère et valide devices audio système
- **Vérifications** : Microphone, speakers, permissions Windows
- **Utilisation** : Diagnostic problèmes audio

#### **`start_llm.py`** - Health-check Serveurs LLM
```bash
python PIPELINE/scripts/start_llm.py
```
- **Fonction** : Teste connectivité serveurs LLM
- **Endpoints** : LM Studio, Ollama, vLLM, llama.cpp
- **Utilisation** : Validation avant démarrage pipeline

#### **`validate_setup.py`** - Validation Complète Setup
```bash
python PIPELINE/scripts/validate_setup.py [--config CONFIG]
```
- **Fonction** : Validation complète environnement + composants
- **Tests** : GPU, audio, LLM, configuration, imports, bootstrap
- **Rapport** : `PIPELINE/reports/validation_setup.json`

---

### **🚀 Scripts de Démonstration**

#### **`demo_pipeline.py`** - Démonstration Interactive
```bash
python PIPELINE/scripts/demo_pipeline.py
```
- **Fonction** : Démonstration pipeline avec code obligatoire du prompt
- **Features** :
  - Menu interactif avec options
  - Démarrage pipeline complet utilisant `_bootstrap()` du prompt
  - Tests validation environnement
  - Affichage configuration
- **Code** : Utilise EXACTEMENT la fonction `_bootstrap()` du prompt

#### **`benchmark_pipeline.py`** - Tests Performance
```bash
python PIPELINE/scripts/benchmark_pipeline.py [--iterations N] [--config CONFIG]
```
- **Fonction** : Benchmark performance composants + pipeline
- **Tests** :
  - STT : Latence transcription
  - LLM : Latence génération
  - TTS : Latence synthèse
  - Pipeline complet : Latence end-to-end
  - Stress test : Requêtes concurrentes
- **Rapport** : `PIPELINE/reports/benchmark_YYYYMMDD_HHMMSS.json`
- **Objectif** : Validation <1.2s end-to-end

---

## 🚨 **CODE OBLIGATOIRE DU PROMPT**

### **Fonction Bootstrap Obligatoire**
Tous les scripts utilisent le code EXACT du prompt v1.1 :

```python
async def _bootstrap(cfg_path: Optional[str] = None):
    """Bootstrap function from prompt - MANDATORY CODE"""
    import yaml
    cfg: Dict[str, Any] = {}
    if cfg_path and Path(cfg_path).exists():
        cfg = yaml.safe_load(Path(cfg_path).read_text())

    # Import components
    from STT.unified_stt_manager_optimized import OptimizedUnifiedSTTManager
    from TTS.tts_manager import UnifiedTTSManager
    from PIPELINE.pipeline_orchestrator import PipelineOrchestrator

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

### **Configuration GPU Obligatoire**
Tous les scripts appliquent la configuration RTX 3090 :

```python
# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
```

---

## 📊 **WORKFLOW RECOMMANDÉ**

### **1. Validation Pré-requis**
```bash
# Validation GPU RTX 3090
python PIPELINE/scripts/assert_gpu_env.py

# Validation audio
python PIPELINE/scripts/validate_audio_devices.py

# Validation LLM
python PIPELINE/scripts/start_llm.py
```

### **2. Validation Setup Complète**
```bash
# Validation complète avec rapport
python PIPELINE/scripts/validate_setup.py
```

### **3. Démonstration Pipeline**
```bash
# Démonstration interactive
python PIPELINE/scripts/demo_pipeline.py
```

### **4. Tests Performance**
```bash
# Benchmark complet
python PIPELINE/scripts/benchmark_pipeline.py --iterations 10
```

---

## 🎯 **OBJECTIFS PERFORMANCE**

### **Métriques Cibles**
- **STT Latence** : <400ms
- **LLM Latence** : <600ms  
- **TTS Latence** : <200ms
- **Pipeline Total** : <1200ms (1.2s)
- **Taux Succès** : >95%

### **Validation Humaine**
- **Tests conversation** : Microphone → réponse vocale
- **Conditions réelles** : Environnement normal utilisateur
- **Critères succès** : Conversation naturelle fluide

---

## 🔧 **DÉPANNAGE**

### **Erreurs Communes**

#### **GPU Non Détectée**
```bash
# Vérifier configuration CUDA
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Relancer validation
python PIPELINE/scripts/assert_gpu_env.py
```

#### **Audio Non Fonctionnel**
```bash
# Vérifier devices audio
python PIPELINE/scripts/validate_audio_devices.py

# Vérifier permissions Windows
# Paramètres → Confidentialité → Microphone
```

#### **LLM Non Accessible**
```bash
# Tester connectivité
python PIPELINE/scripts/start_llm.py

# Vérifier serveur LM Studio/Ollama démarré
curl http://localhost:8000/health
```

### **Logs et Rapports**
- **Validation** : `PIPELINE/reports/validation_setup.json`
- **Benchmark** : `PIPELINE/reports/benchmark_*.json`
- **Logs** : Console avec niveau INFO/DEBUG

---

## 📚 **RÉFÉRENCES**

### **Documentation**
- **Pipeline** : `docs/suivi_pipeline_complet.md`
- **Journal** : `docs/journal_developpement.md`
- **Prompt** : `docs/prompt.md` (code obligatoire v1.1)

### **Configuration**
- **Pipeline** : `PIPELINE/config/pipeline.yaml`
- **Tests** : `PIPELINE/tests/test_pipeline_unit.py`
- **Monitoring** : `PIPELINE/monitoring/`

---

*Scripts Utilitaires SuperWhisper V6*  
*Code Obligatoire Prompt v1.1 - RTX 3090 Exclusif* 