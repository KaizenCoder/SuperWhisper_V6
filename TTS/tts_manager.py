#!/usr/bin/env python3
"""
UnifiedTTSManager - Gestionnaire unifié TTS SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import pathlib

# =============================================================================
# 🚀 PORTABILITÉ AUTOMATIQUE - EXÉCUTABLE DEPUIS N'IMPORTE OÙ
# =============================================================================
def _setup_portable_environment():
    """Configure l'environnement pour exécution portable"""
    # Déterminer le répertoire racine du projet
    current_file = pathlib.Path(__file__).resolve()
    
    # Chercher le répertoire racine (contient .git ou marqueurs projet)
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    # Ajouter le projet root au Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Changer le working directory vers project root
    os.chdir(project_root)
    
    # Configuration GPU RTX 3090 obligatoire
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
    
    print(f"🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
    print(f"📁 Project Root: {project_root}")
    print(f"💻 Working Directory: {os.getcwd()}")
    
    return project_root

# Initialiser l'environnement portable
_PROJECT_ROOT = _setup_portable_environment()

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

# Import utilitaires audio
from TTS.utils_audio import pcm_to_wav, is_valid_wav, get_wav_info

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
        self.model_path = config['model_path']
        self.model_config_path = config['model_config_path']
        self.device = config['device']
        self.speaker_id = config.get('speaker_id', 0)
        
        # Validation RTX 3090 obligatoire
        validate_rtx3090_configuration()
        
        # Initialisation du modèle Piper
        try:
            # Utilisation de l'exécutable piper.exe car piper-python a des problèmes de dépendances
            self.executable_path = "piper/piper.exe"
            if not Path(self.executable_path).exists():
                raise RuntimeError(f"Exécutable Piper non trouvé: {self.executable_path}")
            logging.info(f"Handler Piper Natif (GPU) initialisé avec {self.model_path}")
        except Exception as e:
            logging.error(f"Erreur initialisation PiperNativeHandler: {e}")
            raise

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """Synthèse vocale via Piper natif optimisé GPU"""
        try:
            # Utilisation de l'exécutable piper.exe avec optimisations GPU
            proc = await asyncio.create_subprocess_exec(
                self.executable_path,
                "--model", self.model_path,
                "--speaker", str(self.speaker_id),
                "--output_raw",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate(text.encode('utf-8'))
            
            if proc.returncode != 0:
                raise RuntimeError(f"Piper Native a échoué: {stderr.decode()}")
            
            # 🔧 CORRECTION FORMAT AUDIO - Conversion PCM → WAV
            if not is_valid_wav(stdout):
                logging.debug("PiperNativeHandler: Conversion PCM brut → WAV")
                stdout = pcm_to_wav(
                    pcm_data=stdout,
                    sample_rate=self.config.get('sample_rate', 22050),
                    channels=self.config.get('channels', 1),
                    sampwidth=2
                )
                logging.debug(f"PiperNativeHandler: WAV généré ({len(stdout)} bytes)")
            
            return stdout
            
        except Exception as e:
            logging.error(f"Erreur PiperNativeHandler: {e}")
            raise

class PiperCliHandler(TTSHandler):
    """Handler pour Piper via ligne de commande (CPU) - Optimisé"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.executable_path = config['executable_path']
        self.model_path = config['model_path']
        self.speaker_id = config.get('speaker_id', 0)
        
        # Optimisations performance
        self.use_json_config = config.get('use_json_config', True)
        self.length_scale = config.get('length_scale', 1.0)  # Vitesse synthèse
        
        logging.info(f"Handler Piper CLI (CPU) initialisé - Optimisé pour performance <1000ms")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """Synthèse vocale via Piper CLI optimisée"""
        try:
            # Arguments optimisés pour performance
            args = [
                self.executable_path,
                "--model", self.model_path,
                "--output_raw"
            ]
            
            # Optimisations conditionnelles
            if self.speaker_id > 0:
                args.extend(["--speaker", str(self.speaker_id)])
            
            if speed and speed != 1.0:
                args.extend(["--length_scale", str(speed)])
            elif self.length_scale != 1.0:
                args.extend(["--length_scale", str(self.length_scale)])
            
            # Processus optimisé avec timeout
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Communication avec timeout pour éviter blocages
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(text.encode('utf-8')), 
                    timeout=5.0  # Timeout 5s pour éviter blocages
                )
            except asyncio.TimeoutError:
                proc.kill()
                raise RuntimeError("Piper CLI timeout (>5s)")
            
            if proc.returncode != 0:
                raise RuntimeError(f"Piper CLI a échoué: {stderr.decode()}")
            
            # 🔧 CORRECTION FORMAT AUDIO - Conversion PCM → WAV
            if not is_valid_wav(stdout):
                logging.debug("PiperCliHandler: Conversion PCM brut → WAV")
                stdout = pcm_to_wav(
                    pcm_data=stdout,
                    sample_rate=self.config.get('sample_rate', 22050),
                    channels=self.config.get('channels', 1),
                    sampwidth=2
                )
                logging.debug(f"PiperCliHandler: WAV généré ({len(stdout)} bytes)")
            
            return stdout
            
        except Exception as e:
            logging.error(f"Erreur PiperCliHandler optimisé: {e}")
            raise

class SapiFrenchHandler(TTSHandler):
    """Handler pour Windows SAPI"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.voice_name = config.get('voice_name', 'Microsoft Hortense Desktop')
        self.rate = config.get('rate', 0)
        self.volume = config.get('volume', 100)
        
        # Initialisation SAPI
        try:
            import win32com.client
            self.sapi = win32com.client.Dispatch("SAPI.SpVoice")
            
            # Configuration voix française
            voices = self.sapi.GetVoices()
            for i in range(voices.Count):
                voice = voices.Item(i)
                name = voice.GetDescription()
                if self.voice_name in name or 'french' in name.lower() or 'français' in name.lower():
                    self.sapi.Voice = voice
                    logging.info(f"Voix française SAPI sélectionnée: {name}")
                    break
            
            self.sapi.Rate = self.rate
            self.sapi.Volume = self.volume
            logging.info("Handler SAPI Français initialisé.")
            
        except ImportError:
            logging.error("win32com.client non disponible - SAPI désactivé")
            raise
        except Exception as e:
            logging.error(f"Erreur initialisation SAPI: {e}")
            raise

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """Synthèse vocale via SAPI Windows"""
        try:
            import tempfile
            import wave
            import win32com.client
            
            # Créer fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Synthèse vers fichier WAV
            file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
            file_stream.Open(tmp_path, 3)  # SSFMCreateForWrite
            self.sapi.AudioOutputStream = file_stream
            self.sapi.Speak(text)
            file_stream.Close()
            
            # Lire le fichier WAV généré
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            
            # Nettoyer fichier temporaire
            os.unlink(tmp_path)
            
            return audio_data
            
        except Exception as e:
            logging.error(f"Erreur SapiFrenchHandler: {e}")
            raise

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
                
                # 🔧 VALIDATION FINALE FORMAT AUDIO - Sécurité supplémentaire
                if not is_valid_wav(audio_data):
                    logging.warning(f"UnifiedTTSManager: Format invalide détecté pour {backend_type.value}, conversion forcée")
                    audio_data = pcm_to_wav(
                        pcm_data=audio_data,
                        sample_rate=self.config['advanced'].get('sample_rate', 22050),
                        channels=self.config['advanced'].get('channels', 1)
                    )
                
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

    async def cleanup(self):
        """Nettoyage des ressources"""
        # Nettoyage cache si nécessaire
        self.cache.cache.clear()
        self.cache.current_size = 0
        logging.info("UnifiedTTSManager nettoyé.")

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_configuration() 