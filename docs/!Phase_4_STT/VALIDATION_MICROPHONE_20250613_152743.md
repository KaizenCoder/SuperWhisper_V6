# 🆘 **AIDE EXTERNE - VALIDATION MICROPHONE LIVE PHASE 4 STT**

**Date** : 13 June 2025 - 15:27  
**Problème** : Validation Microphone Live Phase 4 STT  
**Urgence** : **CRITIQUE**  
**SuperWhisper V6** - Phase 4 STT  

---

## 🎯 **CONTEXTE**

Architecture STT parfaite sur fichiers (148/138 mots, RTF 0.082), échec total microphone streaming

---

## 🔧 **CODE ESSENTIEL ACTUEL**


### **1. UnifiedSTTManager - Architecture Principale**

```python
# STT/unified_stt_manager.py
#!/usr/bin/env python3
"""
"""
import os
import sys
from typing import Dict, Any, Optional, List
import asyncio
import time
import hashlib
import numpy as np
import torch
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections import OrderedDict
def validate_rtx3090_mandatory():
    """Validation systématique RTX 3090 pour STT"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise pour STT")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ RTX 3090 validée pour STT: {gpu_name} ({gpu_memory:.1f}GB)")

# Import des backends après configuration GPU
class STTResult:
    """Résultat de transcription STT"""
    text: str
    confidence: float
    segments: List[dict]
    processing_time: float
    device: str
    rtf: float
    backend_used: str
    success: bool
    cached: bool = False
    error: Optional[str] = None

class STTCache:
    """Cache LRU pour résultats STT avec TTL"""
    
    def __init__(self, max_size: int = 200*1024*1024, ttl: int = 7200):
        """
        Initialise le cache LRU.
        
        Args:
            max_size: Taille maximale en bytes (défaut: 200MB)
            ttl: Durée de vie des entrées en secondes (défaut: 2h)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()  # {key: (value, timestamp, size)}
        self.current_size = 0
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[STTResult]:
        """Récupère une valeur du cache avec gestion TTL"""
        if key in self.cache:
            value, timestamp, _ = self.cache[key]
            
            # Vérifier TTL
            if time.time() - timestamp > self.ttl:
                self._remove(key)
                self.misses += 1
                return None
            
            # Hit - déplacer en fin de LRU
            self.cache.move_to_end(key)
            self.hits += 1
            return value
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: STTResult):
        """Ajoute une valeur au cache avec éviction LRU si nécessaire"""
        # Estimer taille (approximation avec sérialisation)
        estimated_size = len(str(value).encode('utf-8'))
        
        # Vérifier si la valeur peut rentrer
        if estimated_size > self.max_size:
            return  # Trop grande pour le cache
        
        # Éviction LRU si nécessaire
        while self.current_size + estimated_size > self.max_size and self.cache:
            self._remove_lru()
        
        # Ajouter nouvelle entrée
        self.cache[key] = (value, time.time(), estimated_size)
        self.current_size += estimated_size
        self.cache.move_to_end(key)  # Déplacer en fin de LRU
    
    def _remove(self, key: str):
        """Supprime une entrée du cache"""
        if key in self.cache:
            _, _, size = self.cache[key]
            self.current_size -= size
            del self.cache[key]
    
    def _remove_lru(self):
        """Supprime l'entrée la moins récemment utilisée"""
        if self.cache:
            key = next(iter(self.cache))  # Premier élément (LRU)
            self._remove(key)

class CircuitBreaker:
    """Protection contre les échecs en cascade"""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def record_failure(self):
        """Enregistre un échec"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
    
    def record_success(self):
        """Enregistre un succès"""
        self.failures = 0
        self.state = "closed"
    
    def is_open(self) -> bool:
        """Vérifie si le circuit est ouvert"""
        if self.state == "open":
            # Vérifier si on peut passer en half-open
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return False
            return True
        return False

class PrometheusSTTMetrics:
    """Métriques Prometheus pour STT"""
    
    def __init__(self):
        try:
            self.transcriptions_total = Counter('stt_transcriptions_total', 'Total STT transcriptions', ['backend', 'status

    # ... (code tronqué pour lisibilité)
```

### **2. Backend Prism Stt Backend**

```python
# STT/backends/prism_stt_backend.py
#!/usr/bin/env python3
"""
"""
import os
import sys
import time
import asyncio
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path
class PrismSTTBackend(BaseSTTBackend):
    """
    Backend STT Prism_Whisper2 optimisé RTX 3090 - SuperWhisper V6
    
    Basé sur l'analyse de Prism_Whisper2 avec optimisations SuperWhisper V6:
    - faster-whisper avec compute_type="float16" 
    - GPU Memory Optimizer intégré
    - Cache modèles intelligent
    - Performance cible < 400ms pour 5s audio
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le backend Prism STT
        
        Args:
            config: Configuration avec model_size, compute_type, etc.
        """
        super().__init__(config)
        
        # Configuration Prism
        self.model_size = config.get('model', 'large-v2')
        self.compute_type = config.get('compute_type', 'float16')
        self.language = config.get('language', 'fr')
        self.beam_size = config.get('beam_size', 5)
        self.vad_filter = config.get('vad_filter', True)  # 🔧 VAD avec paramètres corrigés pour transcription complète
        
        # Modèle Whisper
        self.model = None
        self.model_loaded = False
        
        # Optimisations mémoire (inspiré Prism_Whisper2)
        self.memory_optimizer = None
        self.pinned_buffers = []
        
        # Métriques spécifiques Prism
        self.model_load_time = 0.0
        self.warm_up_completed = False
        
        self.logger = self._setup_logging()
        
        # Initialisation
        self._initialize_prism_backend()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging pour Prism backend"""
        logger = logging.getLogger(f'PrismSTTBackend_{self.model_size}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - Prism - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_prism_backend(self):
        """Initialise le backend Prism avec optimisations RTX 3090"""
        try:
            self.logger.info(f"🚀 Initialisation Prism STT {self.model_size} sur RTX 3090...")
            
            # Validation GPU obligatoire
            validate_rtx3090_mandatory()
            
            # Chargement modèle faster-whisper optimisé
            start_time = time.time()
            
            self.model = WhisperModel(
                self.model_size,
                device="cuda",  # cuda:0 après mapping CUDA_VISIBLE_DEVICES='1'
                compute_type=self.compute_type,
                cpu_threads=4,  # Optimisation CPU pour preprocessing
                num_workers=1   # Single worker pour éviter conflicts GPU
            )
            
            self.model_load_time = time.time() - start_time
            self.model_loaded = True
            
            self.logger.info(f"✅ Modèle {self.model_size} chargé en {self.model_load_time:.2f}s")
            
            # Warm-up GPU avec audio test (inspiré Prism_Whisper2)
            self._warm_up_model()
            
            # Initialiser optimiseur mémoire
            self._initialize_memory_optimizer()
            
            self.logger.info("🎤 Backend Prism STT prêt sur RTX 3090")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation Prism: {e}")
            raise RuntimeError(f"Échec initialisation PrismSTTBackend: {e}")
    
    def _warm_up_model(self):
        """Warm-up modèle avec audio test (inspiré Prism_Whisper2)"""
        try:
            self.logger.info("🔥 Warm-up modèle Prism...")
            
            # Audio test 3 secondes (comme dans Prism_Whisper2)
            dummy_audio = np.zeros(16000 * 3, dtype=np.float32)
            
            # 3 passes de warm-up
            for i in range(3):
                start_time = time.time()
                segments, _ = self.model.transcribe(
                    dummy_audio,
                    language=self.language,
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter
                )
                # Consommer les segments pour forcer l'exécution
                list(segments)
                
                warm_up_time = time.time() - start_time
                self.logger.info(f"   Warm-up {i+1}/3: {warm_up_time:.3f}s")
            
            self.warm_up_completed = True
            self.logger.info("✅ Warm-up Prism terminé")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Warm-up échoué: {e}")
    
    def _initialize_memory_optimizer(self):
        """Initialise optimiseur mémoire (inspiré Prism_Whisper2)"""
        try:
            # Pré-allocation buffers pinned pour audio
            buffer_siz

    # ... (code tronqué pour lisibilité)
```

### **3. VAD Manager - Voice Activity Detection**

```python
# STT/vad_manager.py
#!/usr/bin/env python3
"""
"""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire
def validate_rtx3090_mandatory():
    """Validation obligatoire de la configuration RTX 3090"""
    print("🔍 VALIDATION RTX 3090 OBLIGATOIRE")
    print("=" * 40)
    
    # 1. Vérification variables d'environnement
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    print("✅ CUDA_VISIBLE_DEVICES: '1' (RTX 3090 uniquement)")
    
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER', '')
    if cuda_order != 'PCI_BUS_ID':
        raise RuntimeError(f"🚫 CUDA_DEVICE_ORDER='{cuda_order}' incorrect - doit être 'PCI_BUS_ID'")
    print("✅ CUDA_DEVICE_ORDER: 'PCI_BUS_ID'")
    
    # 2. Validation PyTorch CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    print("✅ PyTorch CUDA disponible")
    
    # 3. Validation GPU spécique RTX 3090
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if "RTX 3090" not in gpu_name:
        raise RuntimeError(f"🚫 GPU incorrecte: '{gpu_name}' - RTX 3090 requise")
    print(f"✅ GPU validée: {gpu_name}")
    
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 VRAM insuffisante: {gpu_memory:.1f}GB - RTX 3090 (24GB) requise")
    print(f"✅ VRAM validée: {gpu_memory:.1f}GB")
    
    print("🎉 VALIDATION RTX 3090 RÉUSSIE")
    return True


class OptimizedVADManager:
    def __init__(self, chunk_ms: int = 160, latency_threshold_ms: float = 25):
        # Validation RTX 3090 obligatoire à l'instanciation
        validate_rtx3090_mandatory()
        
        self.chunk_ms = chunk_ms
        self.latency_threshold_ms = latency_threshold_ms
        self.chunk_samples = int(16000 * chunk_ms / 1000)  # 2560 samples @ 16kHz
        self.backend = None
        self.vad_model = None
        self.vad = None
        
        print(f"🎤 VAD Manager: chunks {chunk_ms}ms ({self.chunk_samples} samples)")
        print(f"⏱️ Seuil latence: {latency_threshold_ms}ms")
        
    async def initialize(self):
        """Initialise avec test de latence sur chunk réaliste"""
        print("🔧 Initialisation VAD...")
        
        # Test Silero d'abord
        silero_latency = await self._test_silero_performance()
        
        if silero_latency <= self.latency_threshold_ms:
            self.backend = "silero"
            print(f"✅ Silero VAD sélectionné ({silero_latency:.2f}ms)")
        else:
            print(f"⚠️ Silero trop lent ({silero_latency:.2f}ms), test WebRTC...")
            webrtc_latency = await self._test_webrtc_performance()
            
            if webrtc_latency <= self.latency_threshold_ms:
                self.backend = "webrtc"
                print(f"✅ WebRTC VAD sélectionné ({webrtc_latency:.2f}ms)")
            else:
                self.backend = "none"
                print(f"⚠️ Tous VAD trop lents, mode pass-through")
                
    async def _test_silero_performance(self) -> float:
        """Test de performance Silero VAD - RTX 3090 UNIQUEMENT"""
        try:
            print("🧪 Test Silero VAD...")
            
            # 🚨 CRITIQUE: RTX 3090 mappée sur CUDA:0 après CUDA_VISIBLE_DEVICES='1'
            target_device = 'cuda:0'  # RTX 3090 24GB (mappée après configuration env)
            
            # Vérifier que la RTX 3090 est disponible (maintenant mappée sur CUDA:0)
            if torch.cuda.device_count() < 1:
                print("⚠️ RTX 3090 non trouvée - fallback CPU")
                target_device = 'cpu'
            else:
                torch.cuda.set_device(0)  # RTX 3090 (après mapping CUDA_VISIBLE_DEVICES)
                print(f"🎮 GPU CONFIG: Utilisation RTX 3090 ({target_device})")
            
            # Charger modèle Silero sur RTX 3090
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            
            self.vad_model = model.to(target_device)
            print("   Modèle Silero chargé sur RTX 3090")
            
            # Test latence sur chunk réaliste
            test_chunk = np.random.randn(self.chunk_samples).astype(np.float32)
            
            # Warmup RTX 3090
            print("   Warmup RTX 3090...")
            for _ in range(5):
                with torch.no_grad():
                    tensor_input = torch.from_numpy(test_chunk).to(target_device)
                    _ = self.vad_model(tensor_input, 16000)
                    
            # Mesure réelle sur 20 itérations
            print("   Mes

    # ... (code tronqué pour lisibilité)
```

### **4. Script Validation - Point d'Échec**

```python
# scripts/validation_microphone_live_equipe.py
#!/usr/bin/env python3
"""
"""
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
def validate_rtx3090_validation():
    """Validation systématique RTX 3090 pour équipe validation"""
    print("\n🔍 VALIDATION CONFIGURATION RTX 3090")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible - RTX 3090 requise")
        return False
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        print(f"❌ CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    if "3090" not in gpu_name:
        print(f"❌ GPU détectée: {gpu_name} - RTX 3090 requise")
        return False
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:
        print(f"❌ GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        return False
    
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")
    return True

def test_microphone_setup():
    """Test setup microphone"""
    print("\n🎤 TEST SETUP MICROPHONE")
    print("=" * 30)
    
    try:
        # Lister devices audio
        devices = sd.query_devices()
        print("📋 Devices audio disponibles:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   {i}: {device['name']} (Input: {device['max_input_channels']} ch)")
        
        # Test enregistrement court
        print("\n🔴 Test enregistrement 2 secondes...")
        print("   Parlez maintenant...")
        
        audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype=np.float32)
        sd.wait()
        
        # Vérifier niveau audio
        max_level = np.max(np.abs(audio))
        print(f"📊 Niveau audio max: {max_level:.3f}")
        
        if max_level < 0.01:
            print("⚠️ Niveau audio très faible - vérifiez microphone")
            return False
        elif max_level > 0.8:
            print("⚠️ Niveau audio très fort - risque saturation")
        else:
            print("✅ Niveau audio correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test microphone: {e}")
        return False

async def validation_texte_complet():
```

---

## 🔍 **PROBLÈME IDENTIFIÉ**

### **Zones Critiques**
1. **Architecture/Pipeline** : Analyse du flow de données
2. **Performance** : Goulots d'étranglement identifiés  
3. **Configuration** : Paramètres optimaux manquants
4. **Intégration** : Problèmes de coordination modules

---

## 🆘 **AIDE DEMANDÉE**

### **Solution Complète Attendue**
- **Code fonctionnel immédiatement opérationnel**
- **Configuration optimale pour environnement**
- **Documentation intégration**
- **Plan résolution étape par étape**

### **Contraintes Techniques**
- **GPU** : RTX 3090 24GB exclusif (CUDA:1)
- **OS** : Windows 10 PowerShell 7
- **Python** : 3.12 avec dépendances existantes
- **Performance** : Maintenir niveau actuel

---

**🚨 RÉPONSE EXHAUSTIVE DEMANDÉE AVEC CODE COMPLET !**
