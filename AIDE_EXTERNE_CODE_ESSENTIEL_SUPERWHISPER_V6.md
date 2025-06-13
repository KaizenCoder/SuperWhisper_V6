# 🆘 **AIDE EXTERNE - CODE ESSENTIEL SUPERWHISPER V6**

**Problème** : Validation microphone live Phase 4 STT  
**Statut** : Architecture STT parfaite sur fichiers, échec total microphone  
**Urgence** : 48-72h maximum  

---

## 🎯 **PROBLÈME RÉSUMÉ**

- ✅ **Tests fichiers audio** : 148/138 mots (107.2%), RTF 0.082 - PARFAIT
- ❌ **Tests microphone live** : 0% réussite - ÉCHEC SYSTÉMATIQUE
- **Hypothèse** : VAD streaming ≠ VAD fichier, ou pipeline async défaillant

---

## 🔧 **CODE ESSENTIEL ACTUEL**

### **1. UnifiedSTTManager - Architecture Principale**

```python
#!/usr/bin/env python3
"""
STT/unified_stt_manager.py
Manager STT unifié avec fallback - FONCTIONNE sur fichiers, PAS sur microphone
"""

import os
import sys
import logging
import asyncio
from typing import Optional, Dict, Any, List
import time

# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

class UnifiedSTTManager:
    """Manager STT unifié avec fallback intelligent"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.backends = []
        self.current_backend = None
        self.metrics = {
            'transcriptions': 0,
            'errors': 0,
            'avg_latency': 0.0,
            'backend_usage': {}
        }
        
        # Configuration VAD - FONCTIONNE sur fichiers
        self.vad_config = {
            "threshold": 0.3,
            "min_speech_duration_ms": 100,
            "max_speech_duration_s": float('inf'),
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 400
        }
        
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialise les backends STT avec fallback"""
        try:
            from STT.backends.prism_stt_backend import PrismSTTBackend
            backend = PrismSTTBackend()
            if backend.is_available():
                self.backends.append(backend)
                self.logger.info("✅ PrismSTTBackend RTX 3090 initialisé")
        except Exception as e:
            self.logger.error(f"❌ PrismSTTBackend échec: {e}")
        
        # Fallbacks...
        self.current_backend = self.backends[0] if self.backends else None
    
    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        """Transcription fichier - FONCTIONNE PARFAITEMENT"""
        if not self.current_backend:
            raise RuntimeError("Aucun backend STT disponible")
        
        start_time = time.time()
        try:
            result = self.current_backend.transcribe_file(audio_path)
            latency = time.time() - start_time
            
            self.metrics['transcriptions'] += 1
            self.metrics['avg_latency'] = (
                (self.metrics['avg_latency'] * (self.metrics['transcriptions'] - 1) + latency) 
                / self.metrics['transcriptions']
            )
            
            return {
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0.0),
                'latency': latency,
                'backend': self.current_backend.__class__.__name__,
                'rtf': latency / result.get('duration', 1.0)
            }
            
        except Exception as e:
            self.metrics['errors'] += 1
            self.logger.error(f"❌ Transcription échec: {e}")
            raise
    
    async def transcribe_stream(self, audio_stream):
        """Transcription streaming - ÉCHEC SYSTÉMATIQUE"""
        # ❌ PROBLÈME ICI - Pipeline streaming défaillant
        if not self.current_backend:
            raise RuntimeError("Aucun backend STT disponible")
        
        try:
            # ❌ Cette méthode échoue systématiquement avec microphone
            async for chunk in audio_stream:
                result = await self.current_backend.transcribe_chunk(chunk)
                if result and result.get('text'):
                    yield result
        except Exception as e:
            self.logger.error(f"❌ Streaming échec: {e}")
            raise
```

### **2. PrismSTTBackend - Backend Principal RTX 3090**

```python
#!/usr/bin/env python3
"""
STT/backends/prism_stt_backend.py
Backend principal Prism_Whisper2 RTX 3090 - PARFAIT fichiers, ÉCHEC microphone
"""

import os
import torch
import logging
from pathlib import Path
import time
import numpy as np

# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

class PrismSTTBackend:
    """Backend Prism_Whisper2 optimisé RTX 3090"""
    
    def __init__(self, model_name="Prism_Whisper2"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.model = None
        self.device = None
        self._validate_rtx3090()
        self._initialize_model()
    
    def _validate_rtx3090(self):
        """Validation RTX 3090 obligatoire"""
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices != '1':
            raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' - doit être '1'")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:
            raise RuntimeError(f"🚫 GPU {gpu_memory:.1f}GB trop petite - RTX 3090 requise")
        
        self.device = torch.device("cuda:1")
        self.logger.info(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    
    def _initialize_model(self):
        """Initialise le modèle Prism_Whisper2"""
        try:
            # Configuration modèle RTX 3090
            self.model = self._load_prism_model()
            self.model.to(self.device)
            self.logger.info("✅ Prism_Whisper2 RTX 3090 initialisé")
        except Exception as e:
            self.logger.error(f"❌ Échec initialisation modèle: {e}")
            raise
    
    def transcribe_file(self, audio_path: str) -> dict:
        """Transcription fichier - FONCTIONNE PARFAITEMENT"""
        if not self.model:
            raise RuntimeError("Modèle non initialisé")
        
        try:
            start_time = time.time()
            
            # Chargement audio
            audio_data = self._load_audio(audio_path)
            
            # Transcription avec RTX 3090
            with torch.no_grad():
                result = self.model.transcribe(audio_data, device=self.device)
            
            latency = time.time() - start_time
            
            return {
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0.0),
                'duration': result.get('duration', 0.0),
                'latency': latency
            }
            
        except Exception as e:
            self.logger.error(f"❌ Transcription fichier échec: {e}")
            raise
    
    async def transcribe_chunk(self, audio_chunk: np.ndarray) -> dict:
        """Transcription chunk streaming - ÉCHEC SYSTÉMATIQUE"""
        # ❌ PROBLÈME ICI - Processing chunk streaming défaillant
        if not self.model:
            raise RuntimeError("Modèle non initialisé")
        
        try:
            # ❌ Cette logique échoue avec audio streaming
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio_chunk).to(self.device)
                result = self.model.transcribe_chunk(audio_tensor)
            
            return {
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0.0),
                'is_final': result.get('is_final', False)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Transcription chunk échec: {e}")
            return None
```

### **3. VAD Manager - Voice Activity Detection**

```python
#!/usr/bin/env python3
"""
STT/vad_manager.py
Voice Activity Detection - PARFAIT fichiers, INSTABLE microphone
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
import webrtcvad

class VADManager:
    """Voice Activity Detection Manager"""
    
    def __init__(self, 
                 threshold: float = 0.3,
                 min_speech_duration_ms: int = 100,
                 max_speech_duration_s: float = float('inf'),
                 min_silence_duration_ms: int = 2000,
                 speech_pad_ms: int = 400):
        
        self.logger = logging.getLogger(__name__)
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        # WebRTC VAD
        self.vad = webrtcvad.Vad(2)  # Aggressivité 2/3
        
        self.logger.info(f"✅ VAD initialisé - threshold: {threshold}")
    
    def process_file(self, audio_path: str) -> List[Tuple[float, float]]:
        """VAD sur fichier complet - FONCTIONNE PARFAITEMENT"""
        try:
            audio_data = self._load_audio(audio_path)
            return self._detect_speech_segments(audio_data)
        except Exception as e:
            self.logger.error(f"❌ VAD fichier échec: {e}")
            return []
    
    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """VAD chunk streaming - INSTABLE"""
        # ❌ PROBLÈME ICI - VAD temps réel instable
        try:
            # Conversion pour WebRTC VAD
            if len(audio_chunk) != int(sample_rate * 0.02):  # 20ms chunks requis
                return False
            
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            is_speech = self.vad.is_speech(audio_bytes, sample_rate)
            
            return is_speech
            
        except Exception as e:
            self.logger.error(f"❌ VAD chunk échec: {e}")
            return False
    
    def _detect_speech_segments(self, audio_data: np.ndarray) -> List[Tuple[float, float]]:
        """Détection segments parole - FONCTIONNE sur fichiers"""
        segments = []
        chunk_duration = 0.02  # 20ms
        sample_rate = 16000
        
        chunk_size = int(sample_rate * chunk_duration)
        speech_chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) == chunk_size:
                is_speech = self.process_chunk(chunk, sample_rate)
                speech_chunks.append((i / sample_rate, is_speech))
        
        # Fusion segments
        current_start = None
        for timestamp, is_speech in speech_chunks:
            if is_speech and current_start is None:
                current_start = timestamp
            elif not is_speech and current_start is not None:
                duration = timestamp - current_start
                if duration >= self.min_speech_duration_ms / 1000:
                    segments.append((current_start, timestamp))
                current_start = None
        
        return segments
```

### **4. Script Validation Microphone - ÉCHEC ACTUEL**

```python
#!/usr/bin/env python3
"""
scripts/validation_microphone_live_equipe.py
Script validation microphone - ÉCHEC SYSTÉMATIQUE
"""

import sounddevice as sd
import numpy as np
import asyncio
import logging
from STT.unified_stt_manager import UnifiedSTTManager

class ValidationMicrophoneLive:
    """Validation microphone live - PROBLÉMATIQUE"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stt_manager = UnifiedSTTManager()
        
        # Configuration audio
        self.sample_rate = 16000
        self.buffer_size = 1024
        self.channels = 1
        
        # Queue audio
        self.audio_queue = asyncio.Queue()
        
    async def start_validation(self):
        """Démarre validation microphone - ÉCHEC"""
        try:
            print("🎤 Démarrage capture microphone...")
            
            # ❌ PROBLÈME ICI - Capture streaming défaillante
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.buffer_size,
                callback=self._audio_callback
            ):
                print("🗣️ Parlez maintenant...")
                
                # ❌ Pipeline streaming défaillant
                async for result in self.stt_manager.transcribe_stream(self._audio_stream()):
                    print(f"Transcription: {result.get('text', '')}")
                    
        except Exception as e:
            self.logger.error(f"❌ Validation microphone échec: {e}")
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback audio - PROBLÉMATIQUE"""
        if status:
            self.logger.warning(f"⚠️ Audio callback status: {status}")
        
        # ❌ Queue peut se saturer ou rester vide
        try:
            self.audio_queue.put_nowait(indata.copy())
        except asyncio.QueueFull:
            self.logger.warning("⚠️ Queue audio saturée")
    
    async def _audio_stream(self):
        """Stream audio - DÉFAILLANT"""
        # ❌ PROBLÈME PRINCIPAL ICI
        while True:
            try:
                audio_chunk = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=1.0
                )
                yield audio_chunk
            except asyncio.TimeoutError:
                self.logger.warning("⚠️ Timeout audio stream")
                break

if __name__ == "__main__":
    validation = ValidationMicrophoneLive()
    asyncio.run(validation.start_validation())
```

---

## 🔍 **ANALYSE PROBLÈME**

### **Zones Suspectes Identifiées**

1. **Pipeline Streaming Audio (scripts/validation_microphone_live_equipe.py)**
   - `_audio_callback()` : Queue saturation/vide
   - `_audio_stream()` : Timeout ou data corruption
   - `sounddevice` configuration inadéquate

2. **VAD Streaming (STT/vad_manager.py)**
   - `process_chunk()` : VAD instable temps réel
   - Chunks 20ms requis vs chunks variables reçus
   - WebRTC VAD inadapté streaming continu

3. **Backend Streaming (STT/backends/prism_stt_backend.py)**
   - `transcribe_chunk()` : Processing défaillant
   - Modèle optimisé fichiers ≠ chunks temps réel
   - GPU memory management streaming

4. **Architecture Async (STT/unified_stt_manager.py)**
   - `transcribe_stream()` : Race conditions
   - Threading/async mal coordonné
   - Gestion erreurs insuffisante

---

## 🆘 **AIDE DEMANDÉE - CODE EXHAUSTIF**

### **Solution Streaming Complète Attendue**

```python
# ✅ SOLUTION DEMANDÉE
class StreamingMicrophoneSTT:
    """Pipeline microphone streaming fonctionnel"""
    
    def __init__(self):
        # Configuration optimale audio streaming
        # VAD streaming temps réel
        # GPU RTX 3090 configuration
        pass
    
    async def start_streaming(self):
        # Pipeline robuste capture → VAD → STT
        # Gestion threading/async optimale
        # Gestion erreurs complète
        pass
    
    def process_audio_realtime(self, audio_chunk):
        # VAD streaming optimisé
        # Transcription chunk temps réel
        # Performance et stabilité
        pass
```

### **Points Critiques à Résoudre**
1. **Audio Capture Streaming** : Configuration sounddevice optimale
2. **VAD Temps Réel** : Algorithme adapté streaming continu
3. **Threading/Async** : Architecture robuste sans race conditions
4. **Memory Management** : GPU RTX 3090 streaming optimisé
5. **Error Handling** : Récupération automatique défaillances

### **Configuration Système**
- **OS** : Windows 10 avec PowerShell 7
- **GPU** : RTX 3090 24GB exclusif (CUDA:1)
- **Microphone** : Rode PodMic USB + microphone intégré
- **Python** : 3.12 avec PyTorch, sounddevice, webrtcvad

---

**🆘 GUIDANCE EXHAUSTIVE DEMANDÉE : Pipeline microphone streaming fonctionnel complet !** 