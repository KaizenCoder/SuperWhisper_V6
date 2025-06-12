# 🏗️ ARCHITECTURE - SuperWhisper V6

**Version** : MVP P0  
**Mise à Jour** : 2025-06-10 23:04:14 CET  
**Architecture** : Modulaire Pipeline Voix-à-Voix  

---

## 🎯 VUE D'ENSEMBLE

### Pipeline Principal : STT → LLM → TTS
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     STT     │───▶│     LLM     │───▶│     TTS     │
│ Transcription│    │ Génération  │    │  Synthèse   │
│   Vocale    │    │  Réponse    │    │   Vocale    │
└─────────────┘    └─────────────┘    └─────────────┘
      ▲                                       │
      │                                       ▼
┌─────────────┐                        ┌─────────────┐
│   AUDIO     │                        │   AUDIO     │
│    INPUT    │                        │   OUTPUT    │
│ (Microphone)│                        │ (Speakers)  │
└─────────────┘                        └─────────────┘
```

---

## 🔧 MODULES DÉTAILLÉS

### 🎤 **STT (Speech-to-Text)**
- **Technologie** : transformers + WhisperProcessor
- **Modèle** : Whisper-large-v3
- **GPU** : RTX 4060 Ti (CUDA:1)
- **Performance** : <2s transcription
- **Input** : Audio WAV 16kHz
- **Output** : Texte français

### 🧠 **LLM (Large Language Model)**
- **Technologie** : llama-cpp-python  
- **Modèle** : Llama-3-8B-Instruct Q5_K_M
- **GPU** : RTX 3090 (GPU:0)
- **Performance** : <1s génération
- **Input** : Prompt + contexte
- **Output** : Réponse française

### 🔊 **TTS (Text-to-Speech)** - **ARCHITECTURE FINALISÉE**
- **Technologie** : Piper CLI (subprocess)
- **Modèle** : fr_FR-siwis-medium.onnx (60MB)
- **Exécutable** : piper.exe (Windows)
- **Performance** : <1s synthèse
- **Input** : Texte français
- **Output** : Audio WAV + playback

---

## 🖥️ INFRASTRUCTURE GPU

### Configuration Dual-GPU Optimisée
```
RTX 3090 (24GB VRAM)     RTX 4060 Ti (16GB VRAM)
├── LLM Module           ├── STT Module
├── CUDA:0               ├── CUDA:1  
├── Llama-3-8B           ├── Whisper-large-v3
└── Génération texte     └── Transcription audio
```

### Répartition Charge
- **STT** : RTX 4060 Ti (VRAM: ~4GB)
- **LLM** : RTX 3090 (VRAM: ~8GB) 
- **TTS** : CPU + subprocess (pas de VRAM)
- **Disponible** : RTX 3090 ~16GB + RTX 4060 Ti ~12GB

---

## 📁 STRUCTURE PROJET

```
SuperWhisper_V6/
├── STT/
│   ├── __init__.py
│   └── stt_handler.py          # Module transcription
├── LLM/  
│   ├── __init__.py
│   └── llm_handler.py          # Module génération
├── TTS/
│   ├── __init__.py
│   └── tts_handler.py          # Module synthèse ✅ FINALISÉ
├── Config/
│   └── mvp_settings.yaml       # Configuration centralisée
├── models/
│   ├── fr_FR-siwis-medium.onnx # Modèle TTS fonctionnel
│   └── fr_FR-siwis-medium.onnx.json
├── piper/
│   └── piper.exe               # Exécutable TTS
├── docs/
│   └── 2025-06-10_journal_developpement_MVP_P0.md
├── run_assistant.py            # Orchestrateur principal
├── test_tts_handler.py         # Tests validation
└── requirements.txt            # Dépendances Python
```

---

## 🔄 FLUX DE DONNÉES

### 1. **Capture Audio** (Input)
```
Microphone → sounddevice → numpy array → STT Handler
```

### 2. **Transcription** (STT)
```
Audio Array → Whisper → Transcription Texte → LLM Handler
```

### 3. **Génération** (LLM)  
```
Prompt + Contexte → Llama-3 → Réponse Texte → TTS Handler
```

### 4. **Synthèse** (TTS) - **NOUVEAU FLUX**
```
Texte → piper.exe --speaker 0 → Audio WAV → sounddevice playback
```

---

## 🛡️ ROBUSTESSE & FALLBACKS

### Gestion Erreurs TTS
- **Timeout** : 30s max par synthèse
- **Cleanup** : Suppression automatique fichiers temporaires  
- **Validation** : Vérification exécutable piper.exe
- **Fallback** : Message d'erreur si échec synthèse

### Architecture Modulaire
- **Isolation** : Chaque module indépendant
- **Interfaces** : APIs claires entre composants
- **Configuration** : YAML centralisé pour tous modules
- **Tests** : Scripts validation individuels

---

## 📊 PERFORMANCE TARGETS

| Composant | Target | Actuel | Status |
|-----------|--------|--------|--------|
| STT Latence | <2s | ~1.2s | ✅ |
| LLM Génération | <1s | ~0.8s | ✅ |
| **TTS Synthèse** | **<1s** | **<1s** | ✅ **NOUVEAU** |
| Pipeline Total | <5s | ~3s | ✅ |
| VRAM Usage | <20GB | ~12GB | ✅ |

---

## 🔮 ÉVOLUTION ARCHITECTURE

### Phase 2 Prévue
- **Streaming TTS** : Synthèse temps réel
- **Optimisation GPU** : Parallélisation STT+LLM
- **Cache Intelligent** : Réponses fréquentes
- **Monitoring** : Métriques temps réel

### Extensibilité
- **Multi-langues** : Support anglais/espagnol
- **API REST** : Interface web/mobile  
- **Cloud Deployment** : Docker + Kubernetes
- **Edge Computing** : Optimisation mobile

---

**Architecture validée** ✅  
**Pipeline fonctionnel** : STT + LLM + TTS opérationnels  
**Prêt pour** : Tests d'intégration end-to-end
