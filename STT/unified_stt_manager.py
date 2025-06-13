#!/usr/bin/env python3
"""
UnifiedSTTManager - SuperWhisper V6 Phase 4
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
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

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 UnifiedSTTManager - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Validation GPU obligatoire
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
try:
    from STT.backends.prism_stt_backend import PrismSTTBackend, STTResult
    from prometheus_client import Counter, Histogram, Gauge
except ImportError as e:
    print(f"⚠️ Import optionnel manquant: {e}")

@dataclass
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
            self.transcriptions_total = Counter('stt_transcriptions_total', 'Total STT transcriptions', ['backend', 'status'])
            self.transcription_latency = Histogram('stt_transcription_latency_seconds', 'STT transcription latency', ['backend'])
            self.cache_hits = Counter('stt_cache_hits_total', 'STT cache hits')
            self.cache_misses = Counter('stt_cache_misses_total', 'STT cache misses')
            self.circuit_breaker_skips = Counter('stt_circuit_breaker_skips_total', 'Circuit breaker skips', ['backend'])
            self.total_failures = Counter('stt_total_failures_total', 'Total STT failures')
            self.transcriptions_success = Counter('stt_transcriptions_success_total', 'Successful transcriptions', ['backend'])
            self.transcriptions_failed = Counter('stt_transcriptions_failed_total', 'Failed transcriptions', ['backend'])
        except ImportError:
            # Métriques factices si Prometheus non disponible
            self.transcriptions_total = self._dummy_metric()
            self.transcription_latency = self._dummy_metric()
            self.cache_hits = self._dummy_metric()
            self.cache_misses = self._dummy_metric()
            self.circuit_breaker_skips = self._dummy_metric()
            self.total_failures = self._dummy_metric()
            self.transcriptions_success = self._dummy_metric()
            self.transcriptions_failed = self._dummy_metric()
    
    def _dummy_metric(self):
        """Métrique factice pour tests"""
        class DummyMetric:
            def inc(self): pass
            def observe(self, value): pass
            def set(self, value): pass
            def labels(self, **kwargs): return self
        return DummyMetric()

class UnifiedSTTManager:
    """Manager STT unifié pour SuperWhisper V6 sur RTX 3090"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le manager STT unifié.
        
        Args:
            config: Configuration optionnelle (sinon utilise les valeurs par défaut)
        """
        # Validation GPU obligatoire
        validate_rtx3090_mandatory()
        
        self.config = config or {
            'cache_size_mb': 200,
            'cache_ttl': 7200,  # 2 heures
            'timeout_per_minute': 5.0,
            'max_retries': 3
        }
        
        # Initialisation des backends avec fallback
        self.backends = {}
        self._initialize_backends()
        
        # Ordre de fallback optimisé RTX 3090
        self.fallback_chain = ['prism_large', 'prism_medium', 'prism_small']
        
        # Cache LRU (cohérent avec TTS Phase 3)
        cache_size = self.config['cache_size_mb'] * 1024 * 1024  # Conversion en bytes
        self.cache = STTCache(max_size=cache_size, ttl=self.config['cache_ttl'])
        
        # Circuit breakers par backend
        self.circuit_breakers = {
            name: CircuitBreaker() for name in self.fallback_chain
        }
        
        # Métriques Prometheus
        self.metrics = PrometheusSTTMetrics()
        
        print("✅ UnifiedSTTManager initialisé sur RTX 3090")
    
    def _initialize_backends(self):
        """Initialise les backends STT avec gestion d'erreur"""
        try:
            # Backend principal Prism
            self.backends['prism_large'] = PrismSTTBackend({
                'model': 'large-v2',
                'compute_type': 'float16'
            })
            
            self.backends['prism_medium'] = PrismSTTBackend({
                'model': 'medium',
                'compute_type': 'float16'
            })
            
            self.backends['prism_small'] = PrismSTTBackend({
                'model': 'small',
                'compute_type': 'float16'
            })
            
            print("✅ Backends STT initialisés")
            
        except Exception as e:
            print(f"⚠️ Erreur initialisation backends: {e}")
            # Fallback minimal pour tests
            self.backends = {
                'prism_large': None,
                'prism_medium': None,
                'prism_small': None
            }
    
    def _generate_cache_key(self, audio: np.ndarray) -> str:
        """
        Génère une clé de cache unique pour l'audio.
        
        Args:
            audio: Données audio numpy
            
        Returns:
            Clé de cache unique
        """
        # Hash MD5 des données audio
        audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
        return f"stt_{audio_hash}_{len(audio)}"
    
    @asynccontextmanager
    async def _memory_management_context(self):
        """Context manager pour la gestion mémoire RTX 3090"""
        try:
            # Nettoyage mémoire avant traitement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield
        finally:
            # Nettoyage mémoire après traitement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def transcribe(self, audio: np.ndarray) -> STTResult:
        """
        Transcrit l'audio en texte avec gestion de cache et fallback.
        
        Args:
            audio: Données audio numpy (16kHz, mono, float32)
            
        Returns:
            Résultat de la transcription avec métriques
            
        Raises:
            Exception: Si tous les backends échouent
        """
        start_time = time.time()
        
        # Vérification cache
        cache_key = self._generate_cache_key(audio)
        if cached_result := self.cache.get(cache_key):
            self.metrics.cache_hits.inc()
            cached_result.cached = True
            return cached_result
        
        self.metrics.cache_misses.inc()
        
        # Calcul timeout dynamique (5s par minute d'audio)
        audio_duration = len(audio) / 16000  # secondes
        timeout = max(5.0, audio_duration * self.config['timeout_per_minute'])
        
        async with self._memory_management_context():
            # Tentative avec chaque backend dans l'ordre
            for backend_name in self.fallback_chain:
                if self.circuit_breakers[backend_name].is_open():
                    self.metrics.circuit_breaker_skips.labels(backend=backend_name).inc()
                    continue
                
                backend = self.backends.get(backend_name)
                if backend is None:
                    continue
                
                try:
                    # Transcription avec timeout
                    result = await asyncio.wait_for(
                        backend.transcribe(audio),
                        timeout=timeout
                    )
                    
                    # Succès - mise en cache et métriques
                    self.cache.put(cache_key, result)
                    self.circuit_breakers[backend_name].record_success()
                    self.metrics.transcriptions_success.labels(backend=backend_name).inc()
                    
                    # Ajout métriques de latence
                    latency = time.time() - start_time
                    self.metrics.transcription_latency.labels(backend=backend_name).observe(latency)
                    
                    result.cached = False
                    return result
                    
                except Exception as e:
                    self.circuit_breakers[backend_name].record_failure()
                    self.metrics.transcriptions_failed.labels(backend=backend_name).inc()
                    print(f"⚠️ Backend {backend_name} échec: {str(e)}")
                    continue
        
        # Tous les backends ont échoué
        self.metrics.total_failures.inc()
        
        # Retourner résultat d'échec
        return STTResult(
            text="",
            confidence=0.0,
            segments=[],
            processing_time=time.time() - start_time,
            device="cuda:0",
            rtf=999.0,
            backend_used="none",
            success=False,
            cached=False,
            error="Tous les backends STT ont échoué"
        )
    
    def get_backend_status(self) -> Dict[str, str]:
        """Retourne le statut des backends"""
        return {
            name: "open" if self.circuit_breakers[name].is_open() else "closed"
            for name in self.fallback_chain
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        total = self.cache.hits + self.cache.misses
        hit_rate = self.cache.hits / total if total > 0 else 0
        
        return {
            "size_bytes": self.cache.current_size,
            "max_size_bytes": self.cache.max_size,
            "usage_percent": (self.cache.current_size / self.cache.max_size) * 100,
            "entries": len(self.cache.cache),
            "hits": self.cache.hits,
            "misses": self.cache.misses,
            "hit_rate": hit_rate
        }
    
    def force_backend(self, backend_name: str) -> None:
        """Force l'utilisation d'un backend spécifique"""
        if backend_name in self.fallback_chain:
            # Réorganiser la chaîne de fallback
            self.fallback_chain = [backend_name] + [b for b in self.fallback_chain if b != backend_name]
            print(f"🔄 Backend forcé: {backend_name}")
        else:
            print(f"⚠️ Backend inconnu: {backend_name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques du système STT.
        
        Returns:
            Dict avec métriques système
        """
        return {
            "cache": self.get_cache_stats(),
            "backends": self.get_backend_status(),
            "gpu": self._get_gpu_metrics() if torch.cuda.is_available() else {}
        }
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """
        Collecte les métriques GPU RTX 3090.
        
        Returns:
            Dict avec métriques GPU
        """
        return {
            "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,  # GB
            "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,    # GB
            "max_memory": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        }

# Point d'entrée pour tests
if __name__ == "__main__":
    print("🧪 Test UnifiedSTTManager")
    
    # Test d'initialisation
    manager = UnifiedSTTManager()
    
    # Test audio factice
    test_audio = np.zeros(16000 * 3, dtype=np.float32)  # 3 secondes
    
    # Test asynchrone
    async def test_transcription():
        result = await manager.transcribe(test_audio)
        print(f"✅ Test transcription: {result.success}")
        print(f"📊 Métriques: {manager.get_metrics()}")
    
    asyncio.run(test_transcription()) 