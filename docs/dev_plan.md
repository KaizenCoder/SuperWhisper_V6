# 📅 PLAN DE DÉVELOPPEMENT - CONSOLIDATION TTS PHASE 2 ENTERPRISE

**Date :** 2025-06-12  
**Version :** v2.0 Enterprise  
**Durée Totale :** 5.5 jours  
**Équipe :** SuperWhisper V6 Core Team  

---

## 🎯 **VUE D'ENSEMBLE STRATÉGIQUE**

### **Philosophie de Développement :**
- **Validation Continue :** Checkpoints bloquants à chaque phase
- **Préservation des Acquis :** Architecture fonctionnelle maintenue
- **Approche Enterprise :** Robustesse + monitoring + performance

### **Architecture Cible :**
```
┌─────────────────────────────────────────────────────────────┐
│                 UnifiedTTSManager                           │
├─────────────────┬─────────────────┬─────────────────────────┤
│ PiperNativeHandler │ PiperCliHandler │ SapiFrenchHandler │ SilentEmergencyHandler │
│   <120ms GPU    │   <1000ms CPU   │   <2000ms SAPI    │     <5ms Silence       │
└─────────────────┴─────────────────┴─────────────────────────┘
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
┌─────────────────┬─────────────────┬─────────────────────────┐
│ Circuit Breakers│   TTSCache      │   Prometheus Metrics    │
│ 3 échecs/30s    │   100MB LRU     │   Temps réel           │
└─────────────────┴─────────────────┴─────────────────────────┘
```

---

## 📋 **PHASE 0 : PRÉPARATION (0.5 JOUR)**

### **🕒 Timing :** J0 - 09h00 → 13h00 (4h)

#### **0.1 - Initialisation Git (1h) :**
```bash
# Création branche feature
git checkout -b feature/tts-enterprise-consolidation

# Tag de sauvegarde
git tag pre-tts-enterprise-consolidation
git push origin pre-tts-enterprise-consolidation

# Validation état initial
git status
git log --oneline -5
```

#### **0.2 - Archivage Handlers Obsolètes (2h) :**
```bash
# Création répertoire archive
mkdir -p TTS/legacy_handlers_20250612

# Documentation rollback
cat > TTS/legacy_handlers_20250612/README_ROLLBACK.md << 'EOF'
# Archive Handlers TTS - 12 juin 2025

## Contexte
Consolidation 15→4 handlers suite Phase 2 Enterprise.
Handlers archivés car non-fonctionnels/redondants.

## Handlers Archivés (13 fichiers)
- tts_handler_piper_native.py (défaillant)
- tts_handler_piper_rtx3090.py (défaillant)
- tts_handler_piper_simple.py (non testé)
- tts_handler_piper_french.py (non testé)
- tts_handler_piper_original.py (legacy)
- tts_handler_piper_direct.py (legacy)
- tts_handler_piper_espeak.py (legacy)
- tts_handler_piper_fixed.py (legacy)
- tts_handler_piper_cli.py (legacy)
- tts_handler_piper.py (legacy)
- tts_handler_coqui.py (alternatif)
- tts_handler_mvp.py (basique)
- tts_handler_fallback.py (interface manquante)

## Rollback Complet
```bash
# Restauration handlers
mv TTS/legacy_handlers_20250612/*.py TTS/
rm -rf TTS/legacy_handlers_20250612/

# Restauration Git
git checkout pre-tts-enterprise-consolidation
git branch -D feature/tts-enterprise-consolidation
```

## Rollback Partiel
```bash
# Restauration handler spécifique
cp TTS/legacy_handlers_20250612/tts_handler_X.py TTS/
```
EOF

# Migration handlers obsolètes
mv TTS/tts_handler_piper_native.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_rtx3090.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_simple.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_french.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_original.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_direct.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_espeak.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_fixed.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_cli.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_coqui.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_mvp.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_fallback.py TTS/legacy_handlers_20250612/
```

#### **0.3 - Script Rollback Automatisé (1h) :**
```bash
# Création script rollback
cat > scripts/rollback_tts_enterprise.sh << 'EOF'
#!/bin/bash
# Script de rollback automatisé TTS Enterprise

echo "🔄 Rollback TTS Enterprise en cours..."

# Vérification tag existe
if ! git tag -l | grep -q "pre-tts-enterprise-consolidation"; then
    echo "❌ Tag de sauvegarde introuvable"
    exit 1
fi

# Sauvegarde état actuel
git stash push -m "Rollback TTS Enterprise $(date)"

# Restauration tag
git checkout pre-tts-enterprise-consolidation

# Nettoyage branche feature
git branch -D feature/tts-enterprise-consolidation 2>/dev/null || true

# Restauration handlers archivés
if [ -d "TTS/legacy_handlers_20250612" ]; then
    mv TTS/legacy_handlers_20250612/*.py TTS/ 2>/dev/null || true
    rm -rf TTS/legacy_handlers_20250612/
fi

# Nettoyage fichiers nouveaux
rm -f config/tts.yaml
rm -f TTS/tts_manager_unified.py
rm -rf TTS/handlers/
rm -rf TTS/components/
rm -f tests/test_unified_tts_manager.py

echo "✅ Rollback TTS Enterprise terminé"
echo "📋 État restauré au tag pre-tts-enterprise-consolidation"
EOF

chmod +x scripts/rollback_tts_enterprise.sh
```

### **✅ Livrables Phase 0 :**
- [x] Branche feature créée
- [x] Tag sauvegarde posé
- [x] 13 handlers archivés
- [x] Documentation rollback
- [x] Script rollback automatisé

---

## 📋 **PHASE 1 : RÉPARATION PIPERNATIVEHANDLER (2 JOURS)**

### **🕒 Timing :** J1-J2 - 09h00 → 17h00 (16h)

#### **1.1 - Diagnostic Handler Défaillant (J1 - 4h) :**

##### **Analyse Erreurs Existantes :**
```python
# Examen tts_handler_piper_native.py archivé
# Identification causes échec :
# - Dépendances manquantes ?
# - Chemin modèle incorrect ?
# - Configuration GPU défaillante ?
# - Interface API obsolète ?
```

##### **Validation Environnement :**
```bash
# Test dépendances
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Test modèles disponibles sur D:\
Get-ChildItem "D:\TTS_Voices\piper" -Name
# ✅ Confirmer : fr_FR-siwis-medium.onnx (63MB) + .json

# Test piper-python
pip list | grep piper
```

##### **Benchmark Baseline :**
```python
# Test handler CLI actuel (référence)
python test_tts_handler.py  # Validation <1000ms
```

#### **1.2 - Implémentation PiperNativeHandler (J1-J2 - 8h) :**

##### **Structure Handler GPU :**
```python
# TTS/handlers/piper_native.py
#!/usr/bin/env python3
"""
PiperNativeHandler - GPU RTX 3090 <120ms
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

import time
import asyncio
from typing import Optional
import torch
import numpy as np

# Import piper-python (à installer)
# from piper import PiperVoice

class PiperNativeHandler:
    def __init__(self, config: dict):
        self.config = config
        self._validate_rtx3090_configuration()
        self._initialize_piper_voice()
        
    def _validate_rtx3090_configuration(self):
        """Validation obligatoire RTX 3090"""
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        device_name = torch.cuda.get_device_name(0)
        if "3090" not in device_name:
            raise RuntimeError(f"🚫 GPU invalide: {device_name} - RTX 3090 requise")
        
        # Allocation VRAM limitée (10% max)
        torch.cuda.set_per_process_memory_fraction(0.1, 0)
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ RTX 3090 validée: {device_name} ({gpu_memory:.1f}GB)")
        
    def _initialize_piper_voice(self):
        """Initialisation voix Piper native"""
        model_path = self.config['model_path']
        # self.voice = PiperVoice.load(model_path, use_cuda=True)
        print(f"✅ PiperNativeHandler initialisé: {model_path}")
        
    async def synthesize(self, text: str, voice: Optional[str] = None, 
                        speed: Optional[float] = None) -> bytes:
        """Synthèse TTS GPU <120ms"""
        start_time = time.perf_counter()
        
        # Synthèse via piper-python GPU
        # audio_bytes = await asyncio.to_thread(self.voice.synthesize, text)
        
        # SIMULATION pour développement
        await asyncio.sleep(0.08)  # Simule 80ms
        audio_bytes = b"fake_gpu_audio_data"
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if latency_ms > 120:
            raise RuntimeError(f"Performance dégradée: {latency_ms:.0f}ms > 120ms")
            
        print(f"✅ PiperNative synthèse: {latency_ms:.0f}ms")
        return audio_bytes
```

#### **1.3 - Tests Performance <120ms (J2 - 4h) :**

##### **Benchmarks Validation :**
```python
# tests/test_piper_native_performance.py
import pytest
import asyncio
import time
from TTS.handlers.piper_native import PiperNativeHandler

@pytest.mark.asyncio
async def test_piper_native_latency():
    """Test latence <120ms obligatoire"""
    config = {
        'model_path': 'models/TTS/fr_FR-siwis-medium.onnx',
        'device': 'cuda:0',
        'target_latency_ms': 120
    }
    
    handler = PiperNativeHandler(config)
    
    # Test 10 synthèses
    latencies = []
    for i in range(10):
        start = time.perf_counter()
        await handler.synthesize(f"Test synthèse numéro {i+1}")
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
    
    # Validation P95 <120ms
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    assert p95_latency < 120, f"P95 latence {p95_latency:.0f}ms > 120ms"
    
    print(f"✅ P95 latence: {p95_latency:.0f}ms")
    print(f"📊 Latences: {[f'{l:.0f}ms' for l in latencies]}")

@pytest.mark.asyncio 
async def test_gpu_memory_usage():
    """Test utilisation VRAM ≤10%"""
    import torch
    
    # Mesure VRAM avant
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated(0)
    
    # Synthèse
    handler = PiperNativeHandler(config)
    await handler.synthesize("Test utilisation mémoire GPU")
    
    # Mesure VRAM après
    memory_after = torch.cuda.memory_allocated(0)
    memory_used_mb = (memory_after - memory_before) / 1024**2
    
    # Validation ≤10% de 24GB = 2.4GB
    assert memory_used_mb <= 2400, f"VRAM utilisée {memory_used_mb:.0f}MB > 2400MB"
    
    print(f"✅ VRAM utilisée: {memory_used_mb:.0f}MB")
```

### **🚨 Checkpoint 1 - PiperNativeHandler :**
- [ ] Handler GPU fonctionnel sans erreur
- [ ] Latence P95 <120ms validée
- [ ] VRAM ≤10% RTX 3090 confirmée
- [ ] Tests automatisés passants
- [ ] **TEST RÉEL OBLIGATOIRE :** `python test_tts_real.py` → Audio généré audible
- [ ] **VALIDATION MANUELLE :** Qualité voix française acceptable
- [ ] **MODÈLES D:\ VALIDÉS :** fr_FR-siwis-medium.onnx (63MB) utilisé

**❌ STOP si échec → Fallback architecture actuelle**

### **✅ Livrables Phase 1 :**
- [x] PiperNativeHandler fonctionnel
- [x] Tests performance <120ms
- [x] Validation VRAM ≤10%
- [x] Benchmarks automatisés

---

## 📋 **PHASE 2 : UNIFIEDTTSMANAGER COMPLET (2 JOURS)**

### **🕒 Timing :** J3-J4 - 09h00 → 17h00 (16h)

#### **2.1 - Configuration YAML Centralisée (J3 - 2h) :**

```yaml
# config/tts.yaml
# Configuration unifiée TTS SuperWhisper V6

# ===================================================================
# SECTION PRINCIPALE
# ===================================================================
enable_piper_native: true

# ===================================================================
# CONFIGURATION BACKENDS
# ===================================================================
backends:
  # Priorité 1: Performance optimale GPU
  piper_native:
    enabled: true
    model_path: "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx"
    model_config_path: "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx.json"
    device: "cuda:0"  # Pointera RTX 3090 après CUDA_VISIBLE_DEVICES
    speaker_id: 0
    target_latency_ms: 120

  # Priorité 2: Fallback robuste CPU
  piper_cli:
    enabled: true
    model_path: "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx"
    executable_path: "piper/piper.exe"
    speaker_id: 0
    target_latency_ms: 1000

  # Priorité 3: Fallback système Windows
  sapi_french:
    enabled: true
    voice_name: "Microsoft Hortense Desktop"
    rate: 0
    volume: 100
    target_latency_ms: 2000

  # Priorité 4: Sécurité ultime
  silent_emergency:
    enabled: true
    log_level: "CRITICAL"
    alert_webhook: null
    target_latency_ms: 5

# ===================================================================
# COMPOSANTS ROBUSTESSE
# ===================================================================
cache:
  enabled: true
  max_size_mb: 100
  ttl_seconds: 3600
  eviction_policy: "LRU"

circuit_breaker:
  failure_threshold: 3
  reset_timeout_seconds: 30

monitoring:
  prometheus_enabled: true
  prometheus_port: 9090
  log_performance_metrics: true
  alert_on_fallback: true

# ===================================================================
# PARAMÈTRES AVANCÉS
# ===================================================================
advanced:
  gpu_memory_fraction: 0.1
  async_workers: 2
  max_text_length: 1000
  sanitize_text: true

# ===================================================================
# FEATURE FLAGS
# ===================================================================
feature_flags:
  use_unified_tts: true
  enable_legacy_mode: false
```

#### **2.2 - Composants Robustesse (J3 - 6h) :**

##### **Circuit Breaker :**
```python
# TTS/components/circuit_breaker.py
import time
import logging
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Isolation handlers défaillants"""
    
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        
    def is_open(self) -> bool:
        """Vérifie si circuit ouvert"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                logging.info("Circuit breaker passage en semi-ouvert")
                return False
            return True
        return False
        
    def record_success(self):
        """Enregistre succès"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logging.info("Circuit breaker refermé")
            
    def record_failure(self):
        """Enregistre échec"""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
                logging.warning(f"Circuit breaker ouvert pour {self.reset_timeout}s")
```

##### **Cache LRU :**
```python
# TTS/components/cache.py
import hashlib
import time
from typing import Dict, Optional, Any
from collections import OrderedDict

class TTSCache:
    """Cache LRU pour synthèses fréquentes"""
    
    def __init__(self, max_size_mb: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size_mb * 1024 * 1024  # Conversion MB
        self.ttl = ttl_seconds
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.current_size = 0
        
    def generate_key(self, text: str, config: Dict) -> str:
        """Génère clé cache"""
        key_str = f"{text}_{config.get('voice', 'default')}_{config.get('speed', 1.0)}"
        return hashlib.sha256(key_str.encode()).hexdigest()
        
    async def get(self, key: str) -> Optional[bytes]:
        """Récupère du cache"""
        if key in self.cache:
            entry = self.cache[key]
            # Vérification TTL
            if time.time() - entry['timestamp'] < self.ttl:
                # Déplacement en fin (LRU)
                self.cache.move_to_end(key)
                return entry['audio_data']
            else:
                # Expiration TTL
                self._remove_entry(key)
        return None
        
    async def set(self, key: str, audio_data: bytes):
        """Stocke en cache"""
        size = len(audio_data)
        
        # Éviction si nécessaire
        while self.current_size + size > self.max_size and self.cache:
            self._evict_lru()
            
        # Stockage
        if self.current_size + size <= self.max_size:
            self.cache[key] = {
                'audio_data': audio_data,
                'timestamp': time.time(),
                'size': size
            }
            self.current_size += size
            
    def _evict_lru(self):
        """Éviction LRU"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.current_size -= entry['size']
            
    def _remove_entry(self, key: str):
        """Supprime entrée"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry['size']
```

#### **2.3 - UnifiedTTSManager Principal (J4 - 8h) :**

```python
# TTS/tts_manager_unified.py
#!/usr/bin/env python3
"""
UnifiedTTSManager - Gestionnaire TTS Enterprise
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

import asyncio
import time
import logging
import yaml
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List
from pathlib import Path

from TTS.handlers.piper_native import PiperNativeHandler
from TTS.handlers.piper_cli import PiperCliHandler  
from TTS.handlers.sapi_french import SapiFrenchHandler
from TTS.handlers.silent_emergency import SilentEmergencyHandler
from TTS.components.circuit_breaker import CircuitBreaker
from TTS.components.cache import TTSCache

class TTSBackendType(Enum):
    PIPER_NATIVE = "piper_native"
    PIPER_CLI = "piper_cli"
    SAPI_FRENCH = "sapi_french"
    SILENT_EMERGENCY = "silent_emergency"
    CACHE = "cache"

@dataclass
class TTSResult:
    success: bool
    backend_used: str
    latency_ms: float
    audio_data: Optional[bytes] = None
    error: Optional[str] = None

class UnifiedTTSManager:
    """Gestionnaire TTS Enterprise avec fallback 4 niveaux"""
    
    def __init__(self, config_path: str = "config/tts.yaml"):
        self.config = self._load_config(config_path)
        self._validate_rtx3090_configuration()
        
        # Initialisation composants
        self.cache = TTSCache(
            max_size_mb=self.config['cache']['max_size_mb'],
            ttl_seconds=self.config['cache']['ttl_seconds']
        )
        
        cb_config = self.config['circuit_breaker']
        self.circuit_breakers = {
            backend: CircuitBreaker(
                failure_threshold=cb_config['failure_threshold'],
                reset_timeout=cb_config['reset_timeout_seconds']
            )
            for backend in TTSBackendType
        }
        
        self.handlers: Dict[TTSBackendType, Any] = {}
        self._initialize_handlers()
        
        logging.info("✅ UnifiedTTSManager initialisé")
        
    def _load_config(self, config_path: str) -> dict:
        """Charge configuration YAML"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def _validate_rtx3090_configuration(self):
        """Validation GPU RTX 3090 obligatoire"""
        import torch
        
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
            
        device_name = torch.cuda.get_device_name(0)
        if "3090" not in device_name:
            raise RuntimeError(f"🚫 GPU invalide: {device_name} - RTX 3090 requise")
            
        # Allocation VRAM limitée
        gpu_mem_fraction = self.config['advanced']['gpu_memory_fraction']
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, 0)
        
        logging.info(f"✅ RTX 3090 validée: {device_name}")
        
    def _initialize_handlers(self):
        """Initialise handlers selon configuration"""
        handler_map = {
            TTSBackendType.PIPER_NATIVE: PiperNativeHandler,
            TTSBackendType.PIPER_CLI: PiperCliHandler,
            TTSBackendType.SAPI_FRENCH: SapiFrenchHandler,
            TTSBackendType.SILENT_EMERGENCY: SilentEmergencyHandler
        }
        
        for backend_type, handler_class in handler_map.items():
            backend_name = backend_type.value
            backend_config = self.config['backends'].get(backend_name, {})
            
            if backend_config.get('enabled', False):
                try:
                    # Vérification feature flag pour piper_native
                    if (backend_type == TTSBackendType.PIPER_NATIVE and 
                        not self.config.get('enable_piper_native', True)):
                        continue
                        
                    self.handlers[backend_type] = handler_class(backend_config)
                    logging.info(f"✅ Handler {backend_name} initialisé")
                    
                except Exception as e:
                    logging.error(f"❌ Échec initialisation {backend_name}: {e}")
                    
    async def synthesize(self, text: str, voice: Optional[str] = None,
                        speed: Optional[float] = None, 
                        reuse_cache: bool = True) -> TTSResult:
        """
        Synthèse TTS unifiée avec fallback automatique
        
        Args:
            text: Texte à synthétiser
            voice: Voix optionnelle
            speed: Vitesse optionnelle
            reuse_cache: Utiliser cache si disponible
            
        Returns:
            TTSResult avec succès, backend utilisé, latence et audio
        """
        start_time_total = time.perf_counter()
        
        # 1. Validation input
        max_len = self.config['advanced']['max_text_length']
        if not text or len(text) > max_len:
            return TTSResult(
                success=False, 
                backend_used="none", 
                latency_ms=0,
                error=f"Texte invalide (vide ou > {max_len} chars)"
            )
            
        # 2. Vérification cache
        cache_key = self.cache.generate_key(text, {'voice': voice, 'speed': speed})
        if reuse_cache and (cached_audio := await self.cache.get(cache_key)):
            latency_ms = (time.perf_counter() - start_time_total) * 1000
            return TTSResult(
                success=True,
                backend_used=TTSBackendType.CACHE.value,
                latency_ms=latency_ms,
                audio_data=cached_audio
            )
            
        # 3. Fallback hiérarchique
        backend_priority = {
            TTSBackendType.PIPER_NATIVE: 1,
            TTSBackendType.PIPER_CLI: 2,
            TTSBackendType.SAPI_FRENCH: 3,
            TTSBackendType.SILENT_EMERGENCY: 4
        }
        
        sorted_backends = sorted(
            self.handlers.keys(), 
            key=lambda x: backend_priority[x]
        )
        
        for backend_type in sorted_backends:
            # Vérification circuit breaker
            if self.circuit_breakers[backend_type].is_open():
                logging.warning(f"Circuit breaker ouvert: {backend_type.value}")
                continue
                
            try:
                start_time_handler = time.perf_counter()
                handler = self.handlers[backend_type]
                
                # Synthèse
                audio_data = await handler.synthesize(text, voice, speed)
                latency_ms = (time.perf_counter() - start_time_handler) * 1000
                
                # Succès
                self.circuit_breakers[backend_type].record_success()
                await self.cache.set(cache_key, audio_data)
                
                # Validation performance
                target_latency = self.config['backends'][backend_type.value]['target_latency_ms']
                if latency_ms > target_latency:
                    logging.warning(
                        f"Performance dégradée: {backend_type.value} "
                        f"{latency_ms:.0f}ms > {target_latency}ms"
                    )
                    
                return TTSResult(
                    success=True,
                    backend_used=backend_type.value,
                    latency_ms=latency_ms,
                    audio_data=audio_data
                )
                
            except Exception as e:
                logging.error(f"Échec {backend_type.value}: {e}")
                self.circuit_breakers[backend_type].record_failure()
                continue
                
        # Tous handlers échoué
        return TTSResult(
            success=False,
            backend_used="none",
            latency_ms=0,
            error="Tous backends TTS échoué"
        )
```

### **🚨 Checkpoint 2 - UnifiedTTSManager :**
- [ ] 4 handlers intégrés et fonctionnels
- [ ] Fallback automatique testé
- [ ] Configuration YAML opérationnelle
- [ ] Tests unitaires 100% passants
- [ ] **TEST FALLBACK RÉEL :** `python test_fallback_real.py` → 4 niveaux validés
- [ ] **ÉCOUTE COMPARATIVE :** Audio de chaque backend (piper_native vs piper_cli vs sapi)
- [ ] **BENCHMARK PERFORMANCE :** `python test_performance_real.py` → P95 <120ms confirmé

### **✅ Livrables Phase 2 :**
- [x] Configuration YAML centralisée
- [x] Circuit Breakers + Cache LRU
- [x] UnifiedTTSManager complet
- [x] Tests unitaires + intégration

---

## 📋 **PHASE 3 : DÉPLOIEMENT & VALIDATION (1 JOUR)**

### **🕒 Timing :** J5 - 09h00 → 17h00 (8h)

#### **3.1 - Tests Validation Complète (4h) :**

##### **Tests Programmatiques :**
```python
# tests/test_unified_tts_manager.py
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from TTS.tts_manager_unified import UnifiedTTSManager, TTSResult

@pytest.mark.asyncio
async def test_fallback_automatique():
    """Test fallback automatique complet"""
    manager = UnifiedTTSManager("config/tts.yaml")
    
    # Simulation panne PiperNative
    with patch.object(manager.handlers[TTSBackendType.PIPER_NATIVE], 
                     'synthesize', side_effect=Exception("GPU failed")):
        result = await manager.synthesize("Test fallback")
        
        # Doit utiliser PiperCLI
        assert result.success == True
        assert result.backend_used == "piper_cli"
        assert result.latency_ms < 1000

@pytest.mark.asyncio
async def test_circuit_breaker():
    """Test circuit breaker isolation"""
    manager = UnifiedTTSManager("config/tts.yaml")
    
    # 3 échecs consécutifs
    for i in range(3):
        with patch.object(manager.handlers[TTSBackendType.PIPER_NATIVE],
                         'synthesize', side_effect=Exception("Fail")):
            await manager.synthesize(f"Test échec {i+1}")
    
    # Circuit breaker doit être ouvert
    assert manager.circuit_breakers[TTSBackendType.PIPER_NATIVE].is_open()

@pytest.mark.asyncio
async def test_cache_performance():
    """Test performance cache"""
    manager = UnifiedTTSManager("config/tts.yaml")
    
    # Premier appel (mise en cache)
    result1 = await manager.synthesize("Test cache")
    
    # Deuxième appel (depuis cache)
    result2 = await manager.synthesize("Test cache")
    
    assert result2.backend_used == "cache"
    assert result2.latency_ms < 5  # <5ms depuis cache

@pytest.mark.asyncio
async def test_performance_regression():
    """Test absence régression performance"""
    manager = UnifiedTTSManager("config/tts.yaml")
    
    # Benchmark 10 synthèses
    latencies = []
    for i in range(10):
        result = await manager.synthesize(f"Test performance {i}")
        latencies.append(result.latency_ms)
    
    # P95 doit respecter targets
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    assert p95_latency < 120  # Target PiperNative
```

##### **🎧 Tests Réels Pratiques OBLIGATOIRES :**
```bash
# 1. VALIDATION MODÈLES DISPONIBLES
Get-ChildItem "D:\TTS_Voices\piper" -Name
# ✅ Confirmer : fr_FR-siwis-medium.onnx (63MB) + .json

# 2. TEST FONCTIONNEL RÉEL
python test_tts_real.py
# ✅ Génère 4 fichiers audio dans test_output/
# ✅ Écouter manuellement chaque fichier
# ✅ Valider qualité voix française

# 3. TEST FALLBACK RÉEL  
python test_fallback_real.py
# ✅ Valide 4 niveaux fallback avec audio généré
# ✅ Confirmer basculement automatique

# 4. BENCHMARK PERFORMANCE RÉEL
python test_performance_real.py
# ✅ 10 mesures par cas (court/moyen/long)
# ✅ Validation P95 <120ms pour piper_native
# ✅ Statistiques détaillées

# 5. ÉCOUTE VALIDATION MANUELLE
start test_output\test_1_piper_native.wav
start test_output\test_2_piper_cli.wav
start test_output\test_3_sapi_french.wav
# ✅ Confirmer audio audible et compréhensible
```

##### **🚨 Critères d'Acceptation Pratiques :**
- ✅ **Audio généré audible** : 4 fichiers test écoutés
- ✅ **Qualité voix française** : Compréhensible et naturelle
- ✅ **Performance mesurée** : <120ms P95 confirmé
- ✅ **Fallback fonctionnel** : 4 niveaux testés avec audio
- ✅ **Aucune régression** : Comparaison avant/après

#### **3.2 - Intégration run_assistant.py (2h) :**

```python
# Modification run_assistant.py
from TTS.tts_manager_unified import UnifiedTTSManager

# Remplacement handler TTS
# OLD: from TTS.tts_handler import TTSHandler
# NEW: tts_manager = UnifiedTTSManager("config/tts.yaml")

async def process_tts(text: str):
    """Traitement TTS unifié"""
    result = await tts_manager.synthesize(text)
    
    if result.success:
        # Lecture audio
        play_audio(result.audio_data)
        
        # Métriques
        print(f"✅ TTS: {result.backend_used} ({result.latency_ms:.0f}ms)")
    else:
        print(f"❌ TTS échec: {result.error}")
```

#### **3.3 - Feature Flags & Monitoring (2h) :**

```python
# Feature flag activation progressive
if config['feature_flags']['use_unified_tts']:
    tts_manager = UnifiedTTSManager("config/tts.yaml")
else:
    # Fallback ancien système
    tts_manager = LegacyTTSHandler()

# Métriques Prometheus basiques
from prometheus_client import Counter, Histogram

tts_requests_total = Counter('tts_requests_total', 'Total TTS requests', ['backend', 'status'])
tts_duration_seconds = Histogram('tts_duration_seconds', 'TTS latency', ['backend'])

# Export métriques
def record_tts_metrics(result: TTSResult):
    status = 'success' if result.success else 'error'
    tts_requests_total.labels(backend=result.backend_used, status=status).inc()
    tts_duration_seconds.labels(backend=result.backend_used).observe(result.latency_ms / 1000)
```

### **🚨 Checkpoint 3 - Déploiement :**
- [ ] Feature flag activation réussie
- [ ] Métriques Prometheus fonctionnelles
- [ ] Performance ≥ baseline
- [ ] Archivage sécurisé + rollback testé

### **✅ Livrables Phase 3 :**
- [x] Tests validation 100% passants
- [x] Intégration run_assistant.py
- [x] Feature flags opérationnels
- [x] Monitoring Prometheus basique

---

## 🎖️ **CRITÈRES D'ACCEPTATION FINALE**

### **✅ Performance :**
- [ ] Latence PiperNative <120ms (P95)
- [ ] Latence PiperCLI <1000ms
- [ ] Latence SAPI <2000ms
- [ ] Cache hit <5ms

### **✅ Robustesse :**
- [ ] Disponibilité 99.9% (fallback)
- [ ] Circuit breakers fonctionnels
- [ ] Recovery automatique
- [ ] Monitoring temps réel

### **✅ Qualité Code :**
- [ ] Type hints 100%
- [ ] Docstrings complètes
- [ ] Tests coverage >90%
- [ ] Configuration externalisée

### **✅ Validation Pratique :**
- [ ] **Tests réels exécutés** : test_tts_real.py, test_fallback_real.py, test_performance_real.py
- [ ] **Audio généré audible** : 4 fichiers test écoutés et validés
- [ ] **Qualité voix française** : Compréhensible et acceptable
- [ ] **Performance mesurée** : <120ms P95 pour piper_native confirmé
- [ ] **Fallback testé** : 4 niveaux validés avec audio généré
- [ ] **Modèles D:\ validés** : fr_FR-siwis-medium.onnx (63MB) utilisé

### **✅ Déploiement :**
- [ ] Feature flags opérationnels
- [ ] Rollback script testé
- [ ] Documentation complète
- [ ] Métriques exportées

---

## 📊 **MÉTRIQUES DE SUCCÈS**

### **🎯 KPIs Post-Déploiement :**

#### **Performance :**
- **Latence moyenne** : <120ms (vs <1000ms)
- **P95 latence** : <150ms
- **Cache hit rate** : >80%
- **Throughput** : >10 synthèses/s

#### **Robustesse :**
- **Uptime** : >99.9%
- **MTBF** : >168h
- **MTTR** : <5s
- **Fallback rate** : <1%

#### **Maintenance :**
- **Complexité code** : -87% fichiers
- **Time to fix** : -50%
- **Deployment time** : <5min
- **Rollback time** : <2min

---

## 🚀 **COMMANDES DE DÉMARRAGE**

```bash
# Phase 0 - Préparation
git checkout -b feature/tts-enterprise-consolidation
git tag pre-tts-enterprise-consolidation
./scripts/rollback_tts_enterprise.sh --test

# Phase 1 - PiperNativeHandler
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
Get-ChildItem "D:\TTS_Voices\piper" -Name
pytest tests/test_piper_native_performance.py -v

# Phase 2 - UnifiedTTSManager
pytest tests/test_unified_tts_manager.py -v
python -m TTS.tts_manager_unified --test

# Phase 3 - Tests Réels Pratiques
python test_tts_real.py
python test_fallback_real.py  
python test_performance_real.py
start test_output\test_1_piper_native.wav

# Phase 3 - Déploiement
pytest tests/ -v --cov=TTS --cov-report=html
python run_assistant.py --feature-flag=unified_tts

echo "🚀 Phase 2 Enterprise - Consolidation TTS terminée !"
```

**🎯 Prêt pour implémentation architecture enterprise UnifiedTTSManager !** 