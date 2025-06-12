# 🚀 PROMPT D'IMPLÉMENTATION - PHASE 4 STT SUPERWHISPER V6

**Version :** 4.1 VALIDATIONS HUMAINES  
**Date :** 12 juin 2025  
**Configuration :** RTX 3090 Unique (24GB VRAM)  
**Statut :** PRÊT POUR IMPLÉMENTATION  

---

## 🎯 CONTEXTE PROJET

Vous êtes en charge d'ajouter le module STT à SuperWhisper V6 sur une **configuration RTX 3090 unique** (24GB VRAM), en utilisant **Prism_Whisper2** ([GitHub](https://github.com/KaizenCoder/Prism_whisper2)) pour compléter le pipeline voix-à-voix.

### **🎯 PRISM_WHISPER2 - SOLUTION VALIDÉE**
**Repository :** [https://github.com/KaizenCoder/Prism_whisper2](https://github.com/KaizenCoder/Prism_whisper2)  
**Status :** Phase 1 TERMINÉE avec validation utilisateur ✅  
**Performance :** 4.5s transcription (vs 7-8s baseline) = **-40% latence**  
**Configuration :** **RTX 3090 optimisé** avec faster-whisper  
**Avantages :**
- ✅ **Architecture mature** : Phase 1 complète avec tests utilisateur
- ✅ **RTX 3090 natif** : Optimisations spécifiques à notre GPU
- ✅ **Faster-whisper** : Backend GPU optimisé éprouvé
- ✅ **99+ langues** : Support multilingue complet
- ✅ **100% local** : Aucune dépendance cloud
- ✅ **Interface Windows** : Intégration Talon système native

### ✅ **État Actuel Validé**
- **Phase 3 TTS** : Terminée avec succès exceptionnel (latence 29.5ms)
- **Configuration GPU** : RTX 3090 exclusive via CUDA_VISIBLE_DEVICES='1'
- **Architecture** : UnifiedTTSManager opérationnel avec 4 backends
- **Performance** : Dépasse tous les objectifs (+340% latence cache)

---

## 🚨 **EXIGENCES CRITIQUES OBLIGATOIRES**

### **🔍 VALIDATIONS HUMAINES OBLIGATOIRES**

**RÈGLE ABSOLUE** : Les tests audio au microphone nécessitent **OBLIGATOIREMENT** une validation humaine par écoute manuelle.

#### **📋 Points de Validation Humaine Obligatoire**
1. **Tests Audio Microphone** : Écoute manuelle obligatoire pour tous tests STT avec microphone réel
2. **Tests Pipeline Voice-to-Voice** : Validation humaine obligatoire pour sessions complètes STT→LLM→TTS
3. **Tests Qualité Audio** : Validation humaine obligatoire pour vérifier qualité transcription et synthèse

#### **📋 Points de Validation Technique (Automatisée)**
1. **Validation Performance** : Métriques automatisées (latence, RTF, etc.)
2. **Validation Architecture** : Tests unitaires et intégration automatisés
3. **Validation Configuration** : Tests GPU et environnement automatisés

#### **📝 Protocole Validation Humaine Audio**
```markdown
### VALIDATION HUMAINE AUDIO - [Type Test] - [Date]
**Date :** [Date validation]
**Validateur :** [Nom/Rôle]
**Type de test :** [Microphone/Pipeline/Qualité]

**Tests Audio Réalisés :**
- [ ] Test microphone avec phrase courte (< 5s)
- [ ] Test microphone avec phrase longue (> 10s)
- [ ] Test qualité transcription (précision)
- [ ] Test pipeline complet voice-to-voice
- [ ] Test conditions audio variables (bruit, distance)

**Résultats Écoute Manuelle :**
- **Qualité Transcription :** [Excellent/Bon/Acceptable/Insuffisant]
- **Précision Mots :** [Pourcentage estimé]
- **Fluidité Pipeline :** [Fluide/Acceptable/Saccadé]
- **Qualité Audio Sortie :** [Claire/Acceptable/Dégradée]

**Résultat :** ✅ VALIDÉ / ❌ À CORRIGER / 🔄 VALIDÉ AVEC RÉSERVES
**Commentaires :** [Feedback détaillé sur qualité audio]
**Actions requises :** [Si corrections nécessaires]
```

### **📚 DOCUMENTATION CONTINUE OBLIGATOIRE**

#### **🔄 Mise à Jour Journal de Développement**
**RÈGLE ABSOLUE** : Le fichier `docs/journal_developpement.md` doit être mis à jour **à chaque session de développement**.

- **❌ INTERDIT** : Supprimer le journal de développement
- **✅ OBLIGATOIRE** : Ajouter une entrée pour chaque session
- **✅ OBLIGATOIRE** : Documenter décisions techniques et problèmes rencontrés
- **✅ OBLIGATOIRE** : Tracer les modifications de code et résultats tests

#### **📊 Suivi des Tâches Obligatoire**
**RÈGLE ABSOLUE** : Créer et maintenir un fichier de suivi spécialisé : `docs/suivi_stt_phase4.md`

**Template obligatoire basé sur :**
```markdown
# 📋 SUIVI STT PHASE 4 SUPERWHISPER V6

**Date de début :** [Date]
**Mission :** Intégration STT Prism_Whisper2 avec RTX 3090
**Référence :** docs/prompt.md + docs/prd.md + docs/dev_plan.md

## 🏆 OBJECTIFS PRINCIPAUX
- ✅/🚧/❌ Intégration Prism_Whisper2 sur RTX 3090
- ✅/🚧/❌ UnifiedSTTManager avec fallback
- ✅/🚧/❌ Pipeline voix-à-voix < 730ms
- ✅/🚧/❌ Tests pratiques avec validation humaine

## 📊 PROGRESSION DÉTAILLÉE
### ✅/🚧/❌ JOUR 1 - PoC et Validation
- ✅/🚧/❌ Setup environnement et validation GPU
- ✅/🚧/❌ Backend PrismSTTBackend implémenté
- ✅/🚧/❌ Tests PoC avec audio réel
- ✅/🚧/❌ **VALIDATION HUMAINE** : Tests audio écoute manuelle

### 🧪 VALIDATIONS HUMAINES RÉALISÉES
[Documenter chaque validation humaine avec détails]

### 📝 DÉCISIONS TECHNIQUES
[Tracer toutes décisions importantes]

### ⚠️ PROBLÈMES ET SOLUTIONS
[Documenter obstacles et résolutions]
```

#### **⏰ Fréquence de Mise à Jour**
- **Journal développement** : Fin de chaque session (minimum quotidien)
- **Suivi tâches** : Temps réel à chaque avancement significatif
- **Validations humaines** : Immédiatement après chaque checkpoint

### 🎮 **Configuration GPU Obligatoire**
```python
# CONFIGURATION RTX 3090 - OBLIGATOIRE AVANT IMPORT TORCH
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 Bus PCI 1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre physique stable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation

# RÉSULTAT MAPPING : cuda:0 → RTX 3090 (24GB)
```

---

## 🚀 **CHECKLIST DÉMARRAGE IMMÉDIAT PHASE 4**

### **📋 Jour 1 - PoC et Validation (À faire aujourd'hui)**

#### **1. Setup Environnement**
```bash
# Installation dépendances
pip install prism-whisper2
pip install prometheus-client
pip install sounddevice  # Pour démo live

# Création structure projet
mkdir -p STT/backends
touch STT/backends/prism_stt_backend.py
touch STT/unified_stt_manager.py
touch tests/test_prism_poc.py
```

#### **2. Clonage et Intégration Prism_Whisper2**
```bash
# Cloner Prism_Whisper2 dans répertoire temporaire
cd /tmp
git clone https://github.com/KaizenCoder/Prism_whisper2.git
cd Prism_whisper2

# Analyser structure et composants
ls -la src/
ls -la src/whisper_engine/
ls -la src/core/

# Intégrer dans SuperWhisper V6
cd /path/to/SuperWhisper_V6
mkdir -p STT/backends/prism
cp -r /tmp/Prism_whisper2/src/whisper_engine/ STT/backends/prism/
cp -r /tmp/Prism_whisper2/src/core/ STT/backends/prism/core/

# Installer dépendances Prism_Whisper2
pip install faster-whisper
pip install sounddevice
pip install numpy
```

#### **3. Script Validation GPU RTX 3090**
**Créer :** `scripts/validate_dual_gpu_rtx3090.py`
```python
#!/usr/bin/env python3
"""
Validation configuration RTX 3090 SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (cuda:0) OBLIGATOIRE
"""

import os
import torch

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 Bus PCI 1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 Validation configuration RTX 3090 SuperWhisper V6")
print(f"CUDA devices disponibles: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"GPU 0: {gpu_name}")
    print(f"VRAM: {gpu_memory:.1f}GB")
    
    # Validation RTX 3090
    if "RTX 3090" in gpu_name and gpu_memory > 20:
        print("\n✅ Configuration validée:")
        print(f"   RTX 3090 (cuda:0) - {gpu_memory:.1f}GB VRAM")
        print("   STT + TTS séquentiel sur même GPU")
    else:
        print(f"\n❌ Configuration incorrecte:")
        print(f"   GPU détecté: {gpu_name} ({gpu_memory:.1f}GB)")
        print("   RTX 3090 24GB requise")
        exit(1)
else:
    print("❌ CUDA non disponible")
    exit(1)
```

---

### **🚨 RAPPEL VALIDATION OBLIGATOIRE**
**AVANT CHAQUE IMPLÉMENTATION** : 
1. ✅ Mise à jour journal développement avec session
2. ✅ Mise à jour suivi tâches avec objectifs
3. ✅ Checkpoint validation humaine planifié

---

## 🎯 STRATÉGIE D'INTÉGRATION STT

### **🎯 SOLUTION PRINCIPALE : PRISM_WHISPER2**

#### **1. Intégration Directe Prism_Whisper2**
**Approche recommandée** : Utiliser Prism_Whisper2 comme backend principal STT

**Avantages :**
- ✅ **Validation terrain** : Phase 1 terminée avec feedback utilisateur positif
- ✅ **Performance prouvée** : 4.5s vs 7-8s baseline (-40% latence)
- ✅ **RTX 3090 optimisé** : Configuration identique à SuperWhisper V6
- ✅ **Architecture mature** : Code stable et testé
- ✅ **Faster-whisper** : Backend éprouvé pour production

**Installation :**
```bash
# Cloner Prism_Whisper2
git clone https://github.com/KaizenCoder/Prism_whisper2.git
cd Prism_whisper2

# Intégration dans SuperWhisper V6
cp -r src/whisper_engine/ ../SuperWhisper_V6/STT/backends/prism/
cp -r src/core/ ../SuperWhisper_V6/STT/backends/prism/core/
```

#### **2. Architecture d'Intégration**
```python
# STT/backends/prism_stt_backend.py
from STT.backends.prism.whisper_engine import SuperWhisper2Engine

class PrismSTTBackend:
    def __init__(self):
        self.engine = SuperWhisper2Engine()
        self.validate_rtx3090_mandatory()
    
    async def transcribe(self, audio_data):
        return await self.engine.transcribe_audio(audio_data)
```

### **🛡️ STRATÉGIE FALLBACK MULTI-NIVEAUX**

#### **Fallback Chain Recommandée :**
```
1. 🚀 PrismSTTBackend (Principal)
   ↓ (si échec)
2. 🔄 WhisperDirectBackend (faster-whisper direct)
   ↓ (si échec)  
3. 🆘 WhisperCPUBackend (CPU fallback)
   ↓ (si échec)
4. 🔇 OfflineSTTBackend (reconnaissance locale basique)
```

#### **1. Backend Principal : PrismSTTBackend**
- **Technologie :** Prism_Whisper2 + faster-whisper
- **GPU :** RTX 3090 exclusif
- **Performance :** 4.5s transcription
- **Modèles :** large-v3, medium, small
- **Langues :** 99+ supportées

#### **2. Fallback 1 : WhisperDirectBackend**
- **Technologie :** faster-whisper direct (sans Prism layer)
- **GPU :** RTX 3090 
- **Performance :** ~6-7s transcription
- **Modèles :** medium, small
- **Usage :** Si Prism_Whisper2 indisponible

#### **3. Fallback 2 : WhisperCPUBackend**
- **Technologie :** whisper CPU-only
- **Processeur :** CPU multithread
- **Performance :** ~15-20s transcription
- **Modèles :** small, tiny
- **Usage :** Si GPU indisponible

#### **4. Fallback 3 : OfflineSTTBackend**
- **Technologie :** Windows Speech Recognition API
- **Processeur :** CPU léger
- **Performance :** ~2-3s (qualité limitée)
- **Langues :** Limitées système
- **Usage :** Urgence absolue

### **📊 COMPARAISON SOLUTIONS STT**

| Backend | Performance | Qualité | VRAM | Fiabilité | Recommandation |
|---------|-------------|---------|------|-----------|----------------|
| **PrismSTTBackend** | ⚡⚡⚡⚡⚡ | 🌟🌟🌟🌟🌟 | 6GB | 🛡️🛡️🛡️🛡️🛡️ | **PRINCIPAL** |
| **WhisperDirectBackend** | ⚡⚡⚡⚡ | 🌟🌟🌟🌟 | 4GB | 🛡️🛡️🛡️🛡️ | **FALLBACK 1** |
| **WhisperCPUBackend** | ⚡⚡ | 🌟🌟🌟 | 0GB | 🛡️🛡️🛡️ | **FALLBACK 2** |
| **OfflineSTTBackend** | ⚡⚡⚡ | 🌟🌟 | 0GB | 🛡️🛡️ | **URGENCE** |

## 🎯 MISSION PHASE 4 STT

### **1. Validation Configuration RTX 3090**
```python
def validate_rtx3090_mandatory():
    """Validation RTX 3090 selon standards SuperWhisper V6"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    # Contrôles obligatoires
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError("🚫 CUDA_VISIBLE_DEVICES incorrect")
    
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 3090" not in gpu_name:
        raise RuntimeError(f"🚫 GPU: {gpu_name} - RTX 3090 requise")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:
        raise RuntimeError(f"🚫 VRAM {gpu_memory:.1f}GB insuffisante")
    
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")
```

### **2. Template PrismSTTBackend CORRIGÉ RTX 3090**
**Créer :** `STT/backends/prism_stt_backend.py`

```python
#!/usr/bin/env python3
"""
Backend STT utilisant Prism_Whisper2 - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (cuda:0) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 Bus PCI 1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (cuda:0) forcée")

import asyncio
import time
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional
from prism_whisper2 import PrismWhisper2

def validate_rtx3090_mandatory():
    """Validation RTX 3090 selon standards SuperWhisper V6"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect")
    
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 3090" not in gpu_name:
        raise RuntimeError(f"🚫 GPU: {gpu_name} - RTX 3090 requise")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:
        raise RuntimeError(f"🚫 VRAM {gpu_memory:.1f}GB insuffisante")
    
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")

@dataclass
class STTResult:
    text: str
    confidence: float
    segments: List[dict]
    processing_time: float
    device: str
    rtf: float
    backend_used: str
    success: bool
    error: Optional[str] = None

class PrismSTTBackend:
    """Backend STT Prism pour RTX 3090 - SuperWhisper V6"""
    
    def __init__(self, config: dict):
        validate_rtx3090_mandatory()
        
        self.model_size = config.get('model', 'large-v2')
        self.device = "cuda:0"  # RTX 3090 après mapping CUDA_VISIBLE_DEVICES='1'
        self.compute_type = config.get('compute_type', 'float16')
        
        print(f"🎤 Initialisation Prism {self.model_size} sur RTX 3090 ({self.device})")
        
        # Chargement modèle sur RTX 3090
        self.model = PrismWhisper2.from_pretrained(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        
        print("✅ Backend Prism STT prêt sur RTX 3090")
    
    async def transcribe(self, audio: np.ndarray) -> STTResult:
        """
        Transcription asynchrone RTX 3090 avec calcul RTF
        
        Args:
            audio: Audio 16kHz mono float32
            
        Returns:
            STTResult avec transcription et métriques
        """
        start_time = time.perf_counter()
        audio_duration = len(audio) / 16000  # secondes
        
        try:
            # Transcription dans thread séparé (éviter blocage asyncio)
            result = await asyncio.to_thread(
                self._transcribe_sync,
                audio
            )
            
            processing_time = time.perf_counter() - start_time
            rtf = processing_time / audio_duration
            
            return STTResult(
                text=result['text'],
                confidence=result.get('confidence', 0.95),
                segments=result.get('segments', []),
                processing_time=processing_time,
                device=self.device,
                rtf=rtf,
                backend_used=f"prism_{self.model_size}",
                success=True
            )
            
        except Exception as e:
            return STTResult(
                text="",
                confidence=0.0,
                segments=[],
                processing_time=time.perf_counter() - start_time,
                device=self.device,
                rtf=999.0,
                backend_used=f"prism_{self.model_size}",
                success=False,
                error=str(e)
            )
    
    def _transcribe_sync(self, audio: np.ndarray) -> dict:
        """Transcription synchrone pour thread - RTX 3090"""
        return self.model.transcribe(
            audio,
            language='fr',
            task='transcribe',
            beam_size=5,
            best_of=5,
            vad_filter=True
        )
    
    def get_gpu_memory_usage(self) -> dict:
        """Surveillance mémoire RTX 3090"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved,
                "usage_percent": (reserved / total) * 100
            }
        return {}
```

### **3. Test PoC RTX 3090 CORRIGÉ**
**Créer :** `tests/test_prism_poc.py`

```python
#!/usr/bin/env python3
"""Tests PoC Prism STT - RTX 3090 SuperWhisper V6"""

import os
# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

import pytest
import numpy as np
import asyncio
from STT.backends.prism_stt_backend import PrismSTTBackend, validate_rtx3090_mandatory

def generate_test_audio(duration=5.0, sample_rate=16000):
    """Génère audio test varié pour validation"""
    # Audio de base (silence)
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # Ajouter bip test 440Hz
    freq = 440  # La 440Hz
    t = np.linspace(0, 0.5, int(sample_rate * 0.5))
    bip = 0.3 * np.sin(2 * np.pi * freq * t)
    audio[sample_rate:sample_rate + len(bip)] = bip
    
    # Ajouter bruit léger pour réalisme
    noise = np.random.normal(0, 0.01, len(audio)).astype(np.float32)
    audio += noise
    
    return audio

@pytest.mark.asyncio
async def test_prism_basic_transcription_rtx3090():
    """Test transcription basique RTX 3090"""
    
    # Validation GPU obligatoire
    validate_rtx3090_mandatory()
    
    # Configuration
    config = {
        'model': 'large-v2',
        'compute_type': 'float16'
    }
    
    # Initialisation backend RTX 3090
    backend = PrismSTTBackend(config)
    
    # Audio test (5 secondes)
    audio = generate_test_audio(duration=5.0)
    
    # Transcription sur RTX 3090
    result = await backend.transcribe(audio)
    
    # Validations performance RTX 3090
    assert result.success, f"Transcription échouée: {result.error}"
    assert result.rtf < 1.0, f"RTF {result.rtf:.2f} > 1.0 (pas temps réel)"
    assert result.processing_time < 1.0, f"Trop lent: {result.processing_time:.2f}s"
    assert result.device == "cuda:0", f"Mauvais GPU: {result.device} (doit être cuda:0)"
    
    # Surveillance mémoire RTX 3090
    memory_usage = backend.get_gpu_memory_usage()
    assert memory_usage["usage_percent"] < 90, f"VRAM saturée: {memory_usage['usage_percent']:.1f}%"
    
    print(f"✅ Transcription RTX 3090: '{result.text}'")
    print(f"⏱️  Latence: {result.processing_time*1000:.0f}ms")
    print(f"📊 RTF: {result.rtf:.2f}")
    print(f"🎮 Device: {result.device}")
    print(f"💾 VRAM: {memory_usage['usage_percent']:.1f}%")

@pytest.mark.asyncio
async def test_prism_tiny_rtx3090():
    """Test modèle tiny plus rapide RTX 3090"""
    
    config = {'model': 'tiny', 'compute_type': 'float16'}
    backend = PrismSTTBackend(config)
    
    audio = generate_test_audio(duration=3.0)
    result = await backend.transcribe(audio)
    
    # Tiny doit être plus rapide
    assert result.success
    assert result.rtf < 0.5, f"RTF tiny {result.rtf:.2f} > 0.5"
    assert result.processing_time < 0.5, f"Latence tiny {result.processing_time:.2f}s > 0.5s"
    
    print(f"✅ Prism Tiny RTX 3090:")
    print(f"   RTF: {result.rtf:.2f}")
    print(f"   Latence: {result.processing_time*1000:.0f}ms")

if __name__ == "__main__":
    asyncio.run(test_prism_basic_transcription_rtx3090())
    asyncio.run(test_prism_tiny_rtx3090())
```

### **4. Configuration YAML Corrigée RTX 3090**
**Créer :** `config/stt.yaml`

```yaml
# Configuration STT SuperWhisper V6 - RTX 3090 Unique
stt:
  primary_backend: prism_whisper2_large
  
  backends:
    prism_whisper2_large:
      model: "large-v2"
      device: "cuda:0"  # RTX 3090 après mapping
      compute_type: "float16"
      language: "fr"
      vad_filter: true
      
    prism_whisper2_tiny:
      model: "tiny"
      device: "cuda:0"  # RTX 3090 après mapping
      compute_type: "float16"
      
  performance:
    timeout_per_minute: 5.0
    max_chunk_duration: 30.0
    target_rtf: 0.5
    max_vram_percent: 80  # RTX 3090 24GB
    
  cache:
    enabled: true
    backend: "memory"  # Ou "redis" si disponible
    ttl: 600
    max_size_mb: 200  # Cohérent avec TTS Phase 3
```

### **5. Script Démo Live RTX 3090**
**Créer :** `scripts/demo_stt_live.py`

```python
#!/usr/bin/env python3
"""Demo STT temps réel avec micro - RTX 3090"""

import os
# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import asyncio
import sounddevice as sd
import numpy as np
from STT.backends.prism_stt_backend import PrismSTTBackend, validate_rtx3090_mandatory

async def demo_stt_live_rtx3090():
    """Démo STT live avec micro RTX 3090"""
    
    print("🎤 Démo STT SuperWhisper V6 - RTX 3090")
    
    # Validation GPU
    validate_rtx3090_mandatory()
    
    # Initialisation backend
    config = {'model': 'large-v2', 'compute_type': 'float16'}
    backend = PrismSTTBackend(config)
    
    print("Appuyez sur ENTER pour enregistrer 5s...")
    input()
    
    # Enregistrement micro
    print("🔴 Enregistrement...")
    audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype=np.float32)
    sd.wait()
    
    # Transcription RTX 3090
    print("🎮 Transcription RTX 3090...")
    result = await backend.transcribe(audio.flatten())
    
    # Affichage résultats
    print(f"\n📝 Transcription: '{result.text}'")
    print(f"⚡ Performance: {result.processing_time*1000:.0f}ms (RTF: {result.rtf:.2f})")
    print(f"🎮 GPU: {result.device}")
    print(f"✅ Succès: {result.success}")
    
    # Surveillance mémoire
    memory = backend.get_gpu_memory_usage()
    print(f"💾 VRAM RTX 3090: {memory['usage_percent']:.1f}%")

if __name__ == "__main__":
    asyncio.run(demo_stt_live_rtx3090())
```

### **6. Monitoring Métriques RTX 3090**
**Créer :** `scripts/monitor_stt_realtime.py`

```python
#!/usr/bin/env python3
"""Monitoring STT temps réel - RTX 3090"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Métriques Prometheus spécifiques RTX 3090
stt_requests = Counter('stt_requests_total', 'Total STT requests', ['backend', 'status', 'gpu'])
stt_latency = Histogram('stt_latency_seconds', 'STT latency', ['backend', 'gpu'])
stt_rtf = Gauge('stt_rtf', 'Real-time factor', ['backend', 'gpu'])
gpu_memory_usage = Gauge('gpu_memory_usage_percent', 'GPU memory usage', ['gpu_model'])

def record_stt_metric_rtx3090(result):
    """Enregistrer métriques STT RTX 3090"""
    status = 'success' if result.success else 'failure'
    gpu_label = 'rtx3090'
    
    stt_requests.labels(backend=result.backend_used, status=status, gpu=gpu_label).inc()
    stt_latency.labels(backend=result.backend_used, gpu=gpu_label).observe(result.processing_time)
    stt_rtf.labels(backend=result.backend_used, gpu=gpu_label).set(result.rtf)

def record_gpu_memory_rtx3090(memory_usage):
    """Surveillance mémoire RTX 3090"""
    gpu_memory_usage.labels(gpu_model='rtx3090').set(memory_usage['usage_percent'])

# Démarrer serveur métriques
if __name__ == "__main__":
    start_http_server(9091)
    print("📊 Métriques RTX 3090 disponibles: http://localhost:9091/metrics")
    
    # Garder serveur actif
    while True:
        time.sleep(1)
```

---

## 📊 **MÉTRIQUES DE SUCCÈS PHASE 4 RTX 3090**

### **KPIs à Tracker dès Jour 1**
| Métrique | Cible RTX 3090 | Mesure | Dashboard |
|----------|---------------|---------|-----------|
| **RTF moyen** | < 0.5 | sum(rtf)/count | Grafana |
| **Latence P95** | < 400ms | Percentile 95 | Prometheus |
| **Taux succès** | > 99% | success/total | AlertManager |
| **Cache hit** | > 30% | hits/requests | Redis stats |
| **GPU usage** | < 80% | nvidia-smi | Grafana |
| **VRAM usage** | < 20GB/24GB | torch.cuda | Monitoring |

### **🔥 Quick Wins Jour 1**

#### **1. Benchmark RTF Immédiat**
```bash
# Créer audios test variés
python scripts/generate_test_audio.py --duration 5 --types "silence,speech,noise"

# Lancer benchmark RTX 3090
python tests/benchmark_rtf.py --model large-v2 --samples 10 --gpu rtx3090
```

#### **2. Validation Standards GPU**
```bash
# Tests validation obligatoire
python test_gpu_correct.py
python test_validation_rtx3090_detection.py

# Validation configuration
python scripts/validate_dual_gpu_rtx3090.py
```

---

## 🎯 STANDARDS OBLIGATOIRES

### **1. Configuration GPU**
```python
# OBLIGATOIRE dans TOUS les fichiers STT
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
```

### **2. Validation RTX 3090**
```python
# OBLIGATOIRE au début de chaque classe/fonction GPU
validate_rtx3090_mandatory()
```

### **3. Device Usage**
```python
# OBLIGATOIRE - Utiliser cuda:0 après mapping
device = "cuda:0"  # RTX 3090 après CUDA_VISIBLE_DEVICES='1'
```

### **4. Tests Validation**
```python
# OBLIGATOIRE avant commit
python test_gpu_correct.py
python test_validation_rtx3090_detection.py
```

---

## 🚀 **VALIDATION FINALE CODE INTÉGRÉ**

✅ **Code Corrigé et Conforme** :
- **Configuration RTX 3090** : `cuda:0` après mapping `CUDA_VISIBLE_DEVICES='1'`
- **Validation GPU** : Fonction `validate_rtx3090_mandatory()` systématique
- **Standards SuperWhisper V6** : Respectés dans tous les fichiers
- **Performance Ciblée** : RTF < 1.0, latence < 400ms
- **Monitoring RTX 3090** : Métriques VRAM et utilisation GPU

✅ **Templates Prêts** :
- `PrismSTTBackend` optimisé RTX 3090
- Tests PoC avec validation performance
- Configuration YAML cohérente
- Scripts démo et monitoring

**🎯 AVEC CE CODE CORRIGÉ, IMPLÉMENTEZ UN STT PROFESSIONNEL SUR RTX 3090 !**  
**🚀 ARCHITECTURE COHÉRENTE + PERFORMANCE EXCEPTIONNELLE + STANDARDS SUPERWHISPER V6**

---

## ❓ **QUESTIONS CRITIQUES POUR VALIDATION**

### **🔍 Questions Techniques Prism_Whisper2**

#### **1. Architecture et Compatibilité**
- **Q1 :** Prism_Whisper2 utilise-t-il une architecture compatible avec notre UnifiedTTSManager ?
- **Q2 :** Les interfaces API de Prism_Whisper2 sont-elles asynchrones (async/await) ?
- **Q3 :** Comment Prism_Whisper2 gère-t-il la configuration GPU RTX 3090 ?
- **Q4 :** Y a-t-il des conflits potentiels entre Prism_Whisper2 et notre TTS Phase 3 ?

#### **2. Performance et Ressources**
- **Q5 :** Quelle est la consommation VRAM de Prism_Whisper2 avec modèle large-v3 ?
- **Q6 :** Peut-on utiliser STT et TTS séquentiellement sur la même RTX 3090 ?
- **Q7 :** Les 4.5s de performance incluent-ils le temps de chargement modèle ?
- **Q8 :** Prism_Whisper2 supporte-t-il le streaming audio temps réel ?

#### **3. Intégration et Configuration**
- **Q9 :** Prism_Whisper2 nécessite-t-il Talon ou peut-il fonctionner indépendamment ?
- **Q10 :** Comment adapter la configuration CUDA_VISIBLE_DEVICES='1' dans Prism_Whisper2 ?
- **Q11 :** Les modèles Whisper de Prism_Whisper2 sont-ils compatibles avec notre cache ?
- **Q12 :** Y a-t-il des dépendances système spécifiques à installer ?

### **🛡️ Questions Fallback et Robustesse**

#### **4. Stratégie de Fallback**
- **Q13 :** Si Prism_Whisper2 échoue, comment détecter l'échec rapidement ?
- **Q14 :** Faster-whisper direct peut-il utiliser les mêmes modèles que Prism_Whisper2 ?
- **Q15 :** Windows Speech Recognition API est-elle suffisante comme fallback d'urgence ?
- **Q16 :** Comment gérer la transition entre backends sans interruption utilisateur ?

#### **5. Tests et Validation**
- **Q17 :** Prism_Whisper2 inclut-il une suite de tests automatisés ?
- **Q18 :** Comment valider la qualité de transcription de manière reproductible ?
- **Q19 :** Quels sont les formats audio supportés par Prism_Whisper2 ?
- **Q20 :** Comment tester la robustesse avec différents accents et qualités audio ?

### **📋 Actions Recommandées Avant Implémentation**

#### **🔴 PRIORITÉ CRITIQUE (À faire immédiatement)**
1. **Cloner et analyser** le repository Prism_Whisper2 complet
2. **Tester** Prism_Whisper2 sur notre configuration RTX 3090
3. **Valider** la compatibilité avec CUDA_VISIBLE_DEVICES='1'
4. **Mesurer** la consommation VRAM réelle avec modèle large-v3

#### **🟠 PRIORITÉ HAUTE (Jour 1)**
5. **Créer** un PoC d'intégration Prism_Whisper2 dans SuperWhisper V6
6. **Tester** la coexistence STT + TTS sur même GPU
7. **Valider** les performances 4.5s sur notre environnement
8. **Documenter** les dépendances et configuration requises

#### **🟡 PRIORITÉ MOYENNE (Jour 2-3)**
9. **Implémenter** la stratégie de fallback multi-niveaux
10. **Créer** les tests d'intégration automatisés
11. **Valider** le pipeline complet STT→LLM→TTS
12. **Optimiser** la gestion mémoire GPU

### **🎯 Décision Finale Recommandée**

**RECOMMANDATION :** Procéder avec **Prism_Whisper2 comme solution principale** avec fallback robuste.

**Justification :**
- ✅ **Validation terrain** : Phase 1 terminée avec succès
- ✅ **Performance prouvée** : -40% latence vs baseline
- ✅ **Configuration identique** : RTX 3090 optimisé
- ✅ **Architecture mature** : Code stable et testé
- ✅ **Fallback sécurisé** : Stratégie multi-niveaux

**Prochaine étape :** Cloner le repository et commencer l'analyse technique détaillée.

---

*Prompt d'Implémentation Phase 4 STT - SuperWhisper V6*  
*Version 4.1 - Prism_Whisper2 Intégré*  
*12 Juin 2025* 