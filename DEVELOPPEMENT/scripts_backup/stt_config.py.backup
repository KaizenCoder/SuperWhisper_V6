#!/usr/bin/env python3
"""
Configuration STT principale - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Configuration centralisée pour tous les backends STT
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

@dataclass
class BackendConfig:
    """Configuration pour un backend STT spécifique"""
    name: str
    enabled: bool = True
    priority: int = 1  # Plus bas = plus prioritaire
    config: Dict[str, Any] = field(default_factory=dict)
    fallback_timeout: float = 5.0  # Timeout avant fallback
    health_check_interval: int = 60  # Secondes entre health checks

@dataclass 
class CacheConfig:
    """Configuration du cache LRU STT"""
    enabled: bool = True
    max_size: int = 100  # Nombre max d'entrées
    ttl: int = 3600  # TTL en secondes (1h)
    cache_dir: str = ".cache/stt"
    persist_cache: bool = True

@dataclass
class STTConfig:
    """Configuration principale STT SuperWhisper V6"""
    
    # Configuration GPU RTX 3090 OBLIGATOIRE
    gpu_device: str = "cuda:1"  # RTX 3090 exclusivement
    cuda_visible_devices: str = "1"
    
    # Backends STT avec priorités
    backends: List[BackendConfig] = field(default_factory=lambda: [
        BackendConfig(
            name="prism",
            enabled=True,
            priority=1,  # Principal
            config={
                "model": "large-v2",
                "compute_type": "float16",
                "language": "fr",
                "beam_size": 5,
                "vad_filter": True
            },
            fallback_timeout=3.0
        ),
        BackendConfig(
            name="whisper_direct",
            enabled=True,
            priority=2,  # Fallback 1
            config={
                "model": "medium",
                "compute_type": "float16",
                "language": "fr"
            },
            fallback_timeout=5.0
        ),
        BackendConfig(
            name="whisper_cpu",
            enabled=True,
            priority=3,  # Fallback 2
            config={
                "model": "small",
                "device": "cpu",
                "language": "fr"
            },
            fallback_timeout=10.0
        )
    ])
    
    # Configuration cache
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Performance
    max_audio_duration: float = 30.0  # Secondes max par requête
    target_latency: float = 0.4  # < 400ms objectif
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5  # Échecs avant ouverture circuit
    recovery_timeout: int = 30  # Secondes avant tentative récupération
    
    # Monitoring
    metrics_enabled: bool = True
    log_level: str = "INFO"
    
    # Validation humaine audio
    human_validation_required: bool = True
    validation_sample_rate: float = 0.1  # 10% des requêtes
    
    @classmethod
    def from_env(cls) -> 'STTConfig':
        """Créer configuration depuis variables d'environnement"""
        config = cls()
        
        # GPU Configuration
        config.gpu_device = os.getenv('STT_GPU_DEVICE', config.gpu_device)
        config.cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', config.cuda_visible_devices)
        
        # Performance
        config.target_latency = float(os.getenv('STT_TARGET_LATENCY', config.target_latency))
        config.max_audio_duration = float(os.getenv('STT_MAX_AUDIO_DURATION', config.max_audio_duration))
        
        # Cache
        config.cache.enabled = os.getenv('STT_CACHE_ENABLED', 'true').lower() == 'true'
        config.cache.max_size = int(os.getenv('STT_CACHE_MAX_SIZE', config.cache.max_size))
        config.cache.ttl = int(os.getenv('STT_CACHE_TTL', config.cache.ttl))
        
        # Monitoring
        config.log_level = os.getenv('STT_LOG_LEVEL', config.log_level)
        config.metrics_enabled = os.getenv('STT_METRICS_ENABLED', 'true').lower() == 'true'
        
        return config
    
    def get_backend_config(self, backend_name: str) -> Optional[BackendConfig]:
        """Obtenir configuration d'un backend spécifique"""
        for backend in self.backends:
            if backend.name == backend_name:
                return backend
        return None
    
    def get_enabled_backends(self) -> List[BackendConfig]:
        """Obtenir backends activés triés par priorité"""
        enabled = [b for b in self.backends if b.enabled]
        return sorted(enabled, key=lambda x: x.priority)
    
    def validate(self) -> List[str]:
        """Valider la configuration et retourner erreurs"""
        errors = []
        
        # Validation GPU
        if not self.gpu_device.startswith('cuda'):
            errors.append(f"GPU device '{self.gpu_device}' invalide - doit être 'cuda:1' pour RTX 3090")
        
        if self.cuda_visible_devices != "1":
            errors.append(f"CUDA_VISIBLE_DEVICES '{self.cuda_visible_devices}' invalide - doit être '1' pour RTX 3090")
        
        # Validation backends
        enabled_backends = self.get_enabled_backends()
        if not enabled_backends:
            errors.append("Aucun backend STT activé")
        
        # Validation performance
        if self.target_latency <= 0:
            errors.append(f"Target latency {self.target_latency} invalide - doit être > 0")
        
        if self.max_audio_duration <= 0:
            errors.append(f"Max audio duration {self.max_audio_duration} invalide - doit être > 0")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        return {
            'gpu_device': self.gpu_device,
            'cuda_visible_devices': self.cuda_visible_devices,
            'backends': [
                {
                    'name': b.name,
                    'enabled': b.enabled,
                    'priority': b.priority,
                    'config': b.config,
                    'fallback_timeout': b.fallback_timeout,
                    'health_check_interval': b.health_check_interval
                }
                for b in self.backends
            ],
            'cache': {
                'enabled': self.cache.enabled,
                'max_size': self.cache.max_size,
                'ttl': self.cache.ttl,
                'cache_dir': self.cache.cache_dir,
                'persist_cache': self.cache.persist_cache
            },
            'max_audio_duration': self.max_audio_duration,
            'target_latency': self.target_latency,
            'circuit_breaker_enabled': self.circuit_breaker_enabled,
            'failure_threshold': self.failure_threshold,
            'recovery_timeout': self.recovery_timeout,
            'metrics_enabled': self.metrics_enabled,
            'log_level': self.log_level,
            'human_validation_required': self.human_validation_required,
            'validation_sample_rate': self.validation_sample_rate
        }

# Configuration par défaut
DEFAULT_STT_CONFIG = STTConfig() 