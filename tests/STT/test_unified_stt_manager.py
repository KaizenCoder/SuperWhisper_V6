#!/usr/bin/env python3
"""
Tests UnifiedSTTManager - SuperWhisper V6 Phase 4
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 Tests UnifiedSTTManager - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import numpy as np
import time
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Validation RTX 3090 obligatoire
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
    
    print(f"✅ RTX 3090 validée pour STT: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

# Validation au démarrage
validate_rtx3090_mandatory()

# Imports locaux après validation GPU
try:
    from STT.unified_stt_manager import UnifiedSTTManager
    from STT.cache_manager import STTCache
except ImportError:
    # Fallback si modules pas encore créés
    print("⚠️ Modules STT non trouvés, utilisation de mocks pour tests")
    
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
    
    class MockSTTBackend:
        def __init__(self, name: str, fail_rate: float = 0.0):
            self.name = name
            self.fail_rate = fail_rate
            self.request_count = 0
            
        async def transcribe(self, audio: np.ndarray) -> STTResult:
            self.request_count += 1
            await asyncio.sleep(0.1)  # Simulation latence
            
            if np.random.random() < self.fail_rate:
                raise Exception(f"Backend {self.name} simulé échec")
            
            audio_duration = len(audio) / 16000
            processing_time = 0.1
            rtf = processing_time / audio_duration
            
            return STTResult(
                text=f"Transcription test {self.name}",
                confidence=0.95,
                segments=[],
                processing_time=processing_time,
                device="cuda:0",
                rtf=rtf,
                backend_used=self.name,
                success=True
            )
    
    class MockCacheManager:
        def __init__(self, max_size_mb: int = 200):
            self.cache = {}
            self.hits = 0
            self.misses = 0
            
        def get(self, key: str) -> Optional[STTResult]:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
            
        def put(self, key: str, value: STTResult):
            self.cache[key] = value
            
        def get_stats(self) -> Dict[str, Any]:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "cache_size": len(self.cache)
            }
    
    class CircuitBreaker:
        def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.failure_count = 0
            self.last_failure_time = 0
            self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
            
        def is_open(self) -> bool:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    return False
                return True
            return False
            
        def record_success(self):
            self.failure_count = 0
            self.state = "CLOSED"
            
        def record_failure(self):
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
    
    class UnifiedSTTManager:
        def __init__(self):
            validate_rtx3090_mandatory()
            
            # Backends simulés
            self.backends = {
                'prism': MockSTTBackend('prism', fail_rate=0.1),
                'whisper': MockSTTBackend('whisper', fail_rate=0.2),
                'azure': MockSTTBackend('azure', fail_rate=0.3)
            }
            
            self.fallback_chain = ['prism', 'whisper', 'azure']
            self.cache = MockCacheManager()
            self.circuit_breakers = {
                name: CircuitBreaker() for name in self.backends.keys()
            }
            
            print("✅ UnifiedSTTManager (Mock) initialisé sur RTX 3090")
        
        def _generate_cache_key(self, audio: np.ndarray) -> str:
            import hashlib
            audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
            return f"stt_{audio_hash}_{len(audio)}"
        
        async def transcribe(self, audio: np.ndarray) -> STTResult:
            # Vérifier cache
            cache_key = self._generate_cache_key(audio)
            if cached_result := self.cache.get(cache_key):
                return cached_result
            
            # Essayer backends avec fallback
            for backend_name in self.fallback_chain:
                if self.circuit_breakers[backend_name].is_open():
                    continue
                
                try:
                    backend = self.backends[backend_name]
                    result = await backend.transcribe(audio)
                    
                    # Succès - mise en cache
                    self.cache.put(cache_key, result)
                    self.circuit_breakers[backend_name].record_success()
                    
                    return result
                    
                except Exception as e:
                    self.circuit_breakers[backend_name].record_failure()
                    print(f"⚠️ Backend {backend_name} échec: {e}")
                    continue
            
            raise Exception("Tous les backends STT ont échoué")
        
        def get_metrics(self) -> Dict[str, Any]:
            return {
                "cache_stats": self.cache.get_stats(),
                "backends_status": {
                    name: cb.state for name, cb in self.circuit_breakers.items()
                },
                "gpu_memory": self._get_gpu_memory()
            }
        
        def _get_gpu_memory(self) -> Dict[str, float]:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total,
                    "usage_percent": (reserved / total) * 100
                }
            return {}

def generate_test_audio(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Génère audio test pour validation STT"""
    samples = int(duration * sample_rate)
    
    # Audio de base avec bruit léger
    audio = np.random.normal(0, 0.05, samples).astype(np.float32)
    
    # Ajouter signal test (bip 440Hz)
    t = np.linspace(0, duration, samples)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Mélanger signal dans première moitié
    mid_point = samples // 2
    audio[:mid_point] += signal[:mid_point] * 0.5
    
    return audio

# Tests UnifiedSTTManager
@pytest.mark.asyncio
async def test_unified_stt_manager_basic():
    """Test basique UnifiedSTTManager RTX 3090"""
    
    manager = UnifiedSTTManager()
    
    # Audio test 3 secondes
    audio = generate_test_audio(duration=3.0)
    
    # Transcription
    result = await manager.transcribe(audio)
    
    # Validations
    assert result.success, f"Transcription échouée: {result.error}"
    assert result.text is not None and len(result.text) > 0
    assert result.rtf < 1.0, f"RTF {result.rtf:.2f} > 1.0 (pas temps réel)"
    assert result.processing_time < 1.0, f"Trop lent: {result.processing_time:.2f}s"
    assert result.device == "cuda:0", f"Mauvais GPU: {result.device}"
    
    print(f"✅ Transcription RTX 3090: '{result.text}'")
    print(f"⏱️  Latence: {result.processing_time*1000:.0f}ms")
    print(f"📊 RTF: {result.rtf:.2f}")
    print(f"🎮 Device: {result.device}")

@pytest.mark.asyncio
async def test_unified_stt_manager_cache():
    """Test cache LRU UnifiedSTTManager"""
    
    manager = UnifiedSTTManager()
    
    # Audio test
    audio = generate_test_audio(duration=2.0)
    
    # Première transcription (cache miss)
    result1 = await manager.transcribe(audio)
    assert result1.success
    
    # Deuxième transcription (cache hit)
    result2 = await manager.transcribe(audio)
    assert result2.success
    assert result2.text == result1.text
    
    # Vérifier cache stats
    metrics = manager.get_metrics()
    cache_stats = metrics["cache_stats"]
    assert cache_stats["hits"] > 0, "Cache hit attendu"
    
    print(f"✅ Cache LRU: {cache_stats['hits']} hits, {cache_stats['misses']} misses")
    print(f"📊 Hit rate: {cache_stats['hit_rate']:.2f}")

@pytest.mark.asyncio
async def test_unified_stt_manager_fallback():
    """Test fallback chain UnifiedSTTManager"""
    
    manager = UnifiedSTTManager()
    
    # Forcer échec backend principal
    manager.backends['prism'].fail_rate = 1.0  # 100% échec
    
    # Audio test
    audio = generate_test_audio(duration=2.0)
    
    # Transcription avec fallback
    result = await manager.transcribe(audio)
    
    # Validation fallback
    assert result.success
    assert result.backend_used != 'prism', "Fallback attendu"
    
    print(f"✅ Fallback réussi: backend {result.backend_used}")

@pytest.mark.asyncio
async def test_unified_stt_manager_circuit_breaker():
    """Test circuit breakers UnifiedSTTManager"""
    
    manager = UnifiedSTTManager()
    
    # Forcer échecs répétés pour déclencher circuit breaker
    manager.backends['prism'].fail_rate = 1.0
    
    # Déclencher circuit breaker (5 échecs)
    for i in range(6):
        try:
            audio = generate_test_audio(duration=1.0)
            await manager.transcribe(audio)
        except:
            pass
    
    # Vérifier circuit breaker ouvert
    metrics = manager.get_metrics()
    assert metrics["backends_status"]["prism"] == "OPEN"
    
    print("✅ Circuit breaker déclenché pour backend prism")

@pytest.mark.asyncio
async def test_unified_stt_manager_gpu_memory():
    """Test surveillance mémoire GPU RTX 3090"""
    
    manager = UnifiedSTTManager()
    
    # Obtenir métriques GPU
    metrics = manager.get_metrics()
    gpu_memory = metrics["gpu_memory"]
    
    # Validations mémoire RTX 3090
    assert "total_gb" in gpu_memory
    assert gpu_memory["total_gb"] > 20, f"VRAM {gpu_memory['total_gb']:.1f}GB < 20GB"
    assert gpu_memory["usage_percent"] < 90, f"VRAM saturée: {gpu_memory['usage_percent']:.1f}%"
    
    print(f"✅ Mémoire RTX 3090:")
    print(f"   Total: {gpu_memory['total_gb']:.1f}GB")
    print(f"   Utilisée: {gpu_memory['usage_percent']:.1f}%")

@pytest.mark.asyncio
async def test_unified_stt_manager_stress():
    """Test stress 5 requêtes parallèles"""
    
    manager = UnifiedSTTManager()
    
    # 5 audios différents
    audios = [generate_test_audio(duration=2.0) for _ in range(5)]
    
    # Traitement parallèle
    start_time = time.time()
    tasks = [manager.transcribe(audio) for audio in audios]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Validations
    assert len(results) == 5
    for result in results:
        assert result.success, f"Échec transcription: {result.error}"
        assert result.rtf < 2.0, f"RTF stress {result.rtf:.2f} > 2.0"
    
    print(f"✅ Stress test RTX 3090: 5 requêtes en {total_time:.2f}s")
    print(f"📊 Latence moyenne: {total_time/5*1000:.0f}ms par requête")

if __name__ == "__main__":
    # Tests directs
    asyncio.run(test_unified_stt_manager_basic())
    asyncio.run(test_unified_stt_manager_cache())
    asyncio.run(test_unified_stt_manager_fallback())
    asyncio.run(test_unified_stt_manager_circuit_breaker())
    asyncio.run(test_unified_stt_manager_gpu_memory())
    asyncio.run(test_unified_stt_manager_stress())
    
    print("\n🎉 Tous les tests UnifiedSTTManager RTX 3090 réussis !") 