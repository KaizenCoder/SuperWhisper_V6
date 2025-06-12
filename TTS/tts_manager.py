#!/usr/bin/env python3
"""
UnifiedTTSManager - Gestionnaire unifié TTS SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Maintenant imports normaux...
import asyncio
import hashlib
import time
import logging
import yaml
import io
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any
from pathlib import Path
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DATA CLASSES ET ENUMS ---
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

# --- HANDLERS SPÉCIFIQUES ---
class TTSHandler(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        pass

class PiperNativeHandler(TTSHandler):
    """Handler pour la lib Piper native (GPU)"""
    def __init__(self, config: dict):
        super().__init__(config)
        logging.info("Handler Piper Natif (GPU) initialisé.")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        await asyncio.sleep(0.1) # Simule la latence
        return b"fake_native_audio_data"

class PiperCliHandler(TTSHandler):
    """Handler pour Piper via ligne de commande (CPU)"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.executable_path = config['executable_path']
        self.model_path = config['model_path']
        logging.info("Handler Piper CLI (CPU) initialisé.")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            self.executable_path,
            "--model", self.model_path,
            "--output_raw",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate(text.encode('utf-8'))
        if proc.returncode != 0:
            raise RuntimeError(f"Piper CLI a échoué: {stderr.decode()}")
        return stdout

class SapiFrenchHandler(TTSHandler):
    """Handler pour Windows SAPI"""
    def __init__(self, config: dict):
        super().__init__(config)
        logging.info("Handler SAPI Français initialisé.")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        await asyncio.sleep(1.5)
        return b"fake_sapi_audio_data"

class SilentEmergencyHandler(TTSHandler):
    """Handler d'urgence qui retourne un silence pour éviter un crash."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.log_level = config.get('log_level', 'CRITICAL')
        logging.info("Handler d'Urgence Silencieux initialisé.")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        logging.log(logging.getLevelName(self.log_level),
                    f"TTS EMERGENCY: Tous les backends ont échoué! Texte: '{text[:50]}...'")
        return self._generate_silent_wav()

    def _generate_silent_wav(self, duration_ms: int = 100) -> bytes:
        sample_rate = 22050
        num_samples = int(sample_rate * duration_ms / 1000)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
            wav.writeframes(b'\x00\x00' * num_samples)
        buffer.seek(0)
        return buffer.read()

# --- COMPOSANTS DE ROBUSTESSE ET PERFORMANCE ---
class CircuitBreaker:
    """Isole un service défaillant pour éviter de le surcharger."""
    def __init__(self, failure_threshold: int, reset_timeout: float):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"

    def is_open(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                return False
            return True
        return False

    def record_success(self):
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            logging.info("Circuit breaker est refermé.")

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            if self.state != "open":
                self.state = "open"
                self.last_failure_time = time.time()
                logging.warning(f"Circuit breaker est ouvert pour {self.reset_timeout}s.")

class TTSCache:
    """Cache en mémoire pour les synthèses fréquentes."""
    def __init__(self, config: dict):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = config.get('max_size_mb', 100) * 1024 * 1024
        self.ttl = config.get('ttl_seconds', 3600)
        self.current_size = 0

    def generate_key(self, text: str, config: Dict) -> str:
        key_str = f"{text}_{config.get('voice', 'default')}_{config.get('speed', 1.0)}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get(self, key: str) -> Optional[bytes]:
        entry = self.cache.get(key)
        if entry and (time.time() - entry['timestamp'] < self.ttl):
            return entry['audio_data']
        return None

    async def set(self, key: str, audio_data: bytes):
        size = len(audio_data)
        if self.current_size + size <= self.max_size:
            self.cache[key] = {'audio_data': audio_data, 'timestamp': time.time(), 'size': size}
            self.current_size += size

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices != '1':
            raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    except ImportError:
        print("⚠️ PyTorch non disponible - validation GPU ignorée")

# --- LE MANAGER UNIFIÉ ---
class UnifiedTTSManager:
    """
    Gestionnaire unifié pour la synthèse vocale (Text-to-Speech).
    Orchestre plusieurs backends TTS avec fallback, cache, et monitoring.
    """
    def __init__(self, config: dict):
        self.config = config
        validate_rtx3090_configuration()

        # Initialisation des composants
        self.cache = TTSCache(config['cache'])
        cb_config = config['circuit_breaker']
        self.circuit_breakers = {
            backend: CircuitBreaker(cb_config['failure_threshold'], cb_config['reset_timeout_seconds'])
            for backend in TTSBackendType
        }
        self.handlers: Dict[TTSBackendType, TTSHandler] = {}
        self._initialize_handlers()
        logging.info("UnifiedTTSManager initialisé avec succès.")

    def _initialize_handlers(self):
        handler_map = {
            TTSBackendType.PIPER_NATIVE: PiperNativeHandler,
            TTSBackendType.PIPER_CLI: PiperCliHandler,
            TTSBackendType.SAPI_FRENCH: SapiFrenchHandler,
            TTSBackendType.SILENT_EMERGENCY: SilentEmergencyHandler
        }
        for backend_type, handler_class in handler_map.items():
            backend_name = backend_type.value
            if self.config['backends'].get(backend_name, {}).get('enabled', False):
                try:
                    if backend_type == TTSBackendType.PIPER_NATIVE and not self.config['enable_piper_native']:
                        continue
                    self.handlers[backend_type] = handler_class(self.config['backends'][backend_name])
                except Exception as e:
                    logging.error(f"Impossible d'initialiser le handler {backend_name}: {e}")

    async def synthesize(self, text: str, voice: Optional[str] = None,
                         speed: Optional[float] = None, reuse_cache: bool = True) -> TTSResult:
        """
        Synthétise du texte en audio avec fallback automatique.
        
        Args:
            text: Texte à synthétiser
            voice: Voix optionnelle (si supportée par le backend)
            speed: Vitesse optionnelle (si supportée par le backend)
            reuse_cache: Utiliser le cache si disponible
            
        Returns:
            TTSResult avec audio_data et métriques
        """
        start_time_total = time.perf_counter()
        
        # 1. Validation de l'input
        max_len = self.config['advanced']['max_text_length']
        if not text or len(text) > max_len:
            return TTSResult(success=False, backend_used="none", latency_ms=0, 
                           error=f"Texte invalide (vide ou > {max_len} chars).")

        # 2. Vérification du cache
        cache_key = self.cache.generate_key(text, {'voice': voice, 'speed': speed})
        if reuse_cache and (cached_audio := await self.cache.get(cache_key)):
            latency_ms = (time.perf_counter() - start_time_total) * 1000
            return TTSResult(success=True, backend_used=TTSBackendType.CACHE.value, 
                           latency_ms=latency_ms, audio_data=cached_audio)
        
        # 3. Chaîne de fallback
        backend_priority = {
            TTSBackendType.PIPER_NATIVE: 1,
            TTSBackendType.PIPER_CLI: 2,
            TTSBackendType.SAPI_FRENCH: 3,
            TTSBackendType.SILENT_EMERGENCY: 4
        }
        sorted_backends = sorted(self.handlers.keys(), key=lambda x: backend_priority[x])

        for backend_type in sorted_backends:
            if self.circuit_breakers[backend_type].is_open():
                continue

            try:
                start_time_handler = time.perf_counter()
                handler = self.handlers[backend_type]
                audio_data = await handler.synthesize(text, voice, speed)
                latency_ms = (time.perf_counter() - start_time_handler) * 1000

                self.circuit_breakers[backend_type].record_success()
                await self.cache.set(cache_key, audio_data)

                # Validation performance
                target_latency = self.config['backends'][backend_type.value]['target_latency_ms']
                if latency_ms > target_latency:
                    logging.warning(f"Performance Warning: {backend_type.value} a dépassé sa cible de latence ({latency_ms:.0f}ms > {target_latency}ms).")

                return TTSResult(success=True, backend_used=backend_type.value, 
                               latency_ms=latency_ms, audio_data=audio_data)

            except Exception as e:
                logging.error(f"Échec du backend {backend_type.value}: {e}")
                self.circuit_breakers[backend_type].record_failure()
                continue
        
        # Si tous les backends ont échoué
        return TTSResult(success=False, backend_used="none", latency_ms=0, 
                       error="Tous les backends TTS ont échoué, y compris l'handler d'urgence.")

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_configuration() 