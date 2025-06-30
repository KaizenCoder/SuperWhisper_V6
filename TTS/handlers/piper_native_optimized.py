#!/usr/bin/env python3
"""
Handler Piper Natif Optimisé - SuperWhisper V6 TTS Phase 3
Binding Python direct avec chargement unique en mémoire et asyncio
🚀 Performance cible: <80ms (vs 500ms CLI)

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
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

try:
    # Import du binding Python Piper natif
    import piper
    from piper import PiperVoice
    PIPER_AVAILABLE = True
    print("✅ Binding Python Piper natif disponible")
except ImportError as e:
    PIPER_AVAILABLE = False
    print(f"⚠️ Binding Python Piper non disponible: {e}")
    print("💡 Fallback vers CLI sera utilisé")

from ..utils_audio import pcm_to_wav, is_valid_wav

class PiperNativeOptimizedHandler:
    """
    Handler Piper Natif Optimisé avec binding Python direct
    
    🚀 OPTIMISATIONS PHASE 3:
    - Chargement unique du modèle en mémoire (vs reload CLI)
    - Binding Python natif (vs subprocess CLI)
    - Appels asyncio.to_thread non-bloquants
    - Cache de voix en mémoire
    - Support GPU RTX 3090 optimisé
    
    Performance cible: <80ms (vs 500ms CLI)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config['model_path']
        self.model_config_path = config.get('model_config_path', self.model_path + '.json')
        self.speaker_id = config.get('speaker_id', 0)
        self.device = config.get('device', 'cuda:0')
        self.target_latency_ms = config.get('target_latency_ms', 80)
        
        # État du handler
        self.voice = None
        self.is_loaded = False
        self.load_time_ms = 0
        self.synthesis_count = 0
        
        # Validation et initialisation
        self._validate_configuration()
        self._load_voice_model()
        
        logging.info(f"Handler Piper Natif Optimisé initialisé - Modèle: {Path(self.model_path).name}")
        logging.info(f"Performance cible: <{self.target_latency_ms}ms")
    
    def _validate_configuration(self):
        """Validation de la configuration et des prérequis"""
        if not PIPER_AVAILABLE:
            raise RuntimeError(
                "Binding Python Piper non disponible. "
                "Installez avec: pip install piper-tts"
            )
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modèle Piper introuvable: {self.model_path}")
        
        if not Path(self.model_config_path).exists():
            raise FileNotFoundError(f"Config modèle introuvable: {self.model_config_path}")
        
        logging.debug(f"✅ Configuration validée - Modèle: {self.model_path}")
    
    def _load_voice_model(self):
        """
        Chargement unique du modèle en mémoire
        🚀 OPTIMISATION: Une seule fois au startup vs reload à chaque appel CLI
        """
        start_time = time.perf_counter()
        
        try:
            # Chargement du modèle avec support CUDA
            use_cuda = 'cuda' in self.device
            
            logging.info(f"🔄 Chargement modèle Piper en mémoire...")
            logging.info(f"   Modèle: {self.model_path}")
            logging.info(f"   Config: {self.model_config_path}")
            logging.info(f"   CUDA: {use_cuda} ({self.device})")
            
            # Chargement via binding Python natif
            self.voice = PiperVoice.load(
                model_path=self.model_path,
                config_path=self.model_config_path,
                use_cuda=use_cuda
            )
            
            # Configuration du speaker
            if hasattr(self.voice, 'speaker_id'):
                self.voice.speaker_id = self.speaker_id
            
            self.is_loaded = True
            self.load_time_ms = (time.perf_counter() - start_time) * 1000
            
            logging.info(f"✅ Modèle Piper chargé en {self.load_time_ms:.1f}ms")
            logging.info(f"   Speakers disponibles: {getattr(self.voice, 'num_speakers', 1)}")
            logging.info(f"   Speaker sélectionné: {self.speaker_id}")
            
        except Exception as e:
            logging.error(f"❌ Échec chargement modèle Piper: {e}")
            self.is_loaded = False
            raise RuntimeError(f"Impossible de charger le modèle Piper: {e}")
    
    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """
        Synthèse vocale optimisée avec binding Python natif
        
        🚀 OPTIMISATIONS:
        - Appel en mémoire (vs subprocess CLI)
        - asyncio.to_thread non-bloquant
        - Pas de fichiers temporaires
        - Conversion PCM→WAV intégrée
        
        Args:
            text: Texte à synthétiser
            voice: Voix optionnelle (ignorée, utilise speaker_id)
            speed: Vitesse optionnelle (si supportée)
            
        Returns:
            bytes: Audio WAV complet avec headers
        """
        if not self.is_loaded:
            raise RuntimeError("Modèle Piper non chargé")
        
        if not text or not text.strip():
            raise ValueError("Texte vide fourni")
        
        start_time = time.perf_counter()
        
        try:
            # 🚀 OPTIMISATION: Appel asyncio non-bloquant
            # Le binding Python est synchrone, on l'exécute dans un thread
            audio_data = await asyncio.to_thread(self._synthesize_sync, text, speed)
            
            # Conversion PCM → WAV si nécessaire
            if not is_valid_wav(audio_data):
                logging.debug("PiperNativeOptimized: Conversion PCM → WAV")
                audio_data = pcm_to_wav(
                    pcm_data=audio_data,
                    sample_rate=self.config.get('sample_rate', 22050),
                    channels=self.config.get('channels', 1),
                    sampwidth=2
                )
            
            # Métriques de performance
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.synthesis_count += 1
            
            # Log de performance
            if latency_ms > self.target_latency_ms:
                logging.warning(
                    f"Performance Warning: PiperNativeOptimized a dépassé sa cible "
                    f"({latency_ms:.1f}ms > {self.target_latency_ms}ms)"
                )
            else:
                logging.debug(
                    f"✅ PiperNativeOptimized: {latency_ms:.1f}ms "
                    f"(cible: {self.target_latency_ms}ms)"
                )
            
            logging.info(
                f"PiperNativeOptimized synthèse #{self.synthesis_count}: "
                f"{len(text)} chars → {len(audio_data)} bytes en {latency_ms:.1f}ms"
            )
            
            return audio_data
            
        except Exception as e:
            logging.error(f"Erreur PiperNativeOptimized: {e}")
            raise RuntimeError(f"Échec synthèse Piper natif: {e}")
    
    def _synthesize_sync(self, text: str, speed: Optional[float] = None) -> bytes:
        """
        Synthèse synchrone via binding Python natif
        
        🚀 OPTIMISATION: Appel direct en mémoire vs subprocess CLI
        """
        try:
            # Configuration de la vitesse si supportée
            if speed is not None and hasattr(self.voice, 'length_scale'):
                # length_scale: 1.0 = normal, <1.0 = plus rapide, >1.0 = plus lent
                self.voice.length_scale = 1.0 / speed if speed > 0 else 1.0
            
            # 🚀 SYNTHÈSE DIRECTE EN MÉMOIRE
            # Retourne directement les données audio (PCM ou WAV selon l'implémentation)
            audio_bytes = self.voice.synthesize(text)
            
            if isinstance(audio_bytes, (list, tuple)):
                # Certaines implémentations retournent une liste d'échantillons
                import struct
                audio_bytes = b''.join(struct.pack('<h', sample) for sample in audio_bytes)
            
            return audio_bytes
            
        except Exception as e:
            raise RuntimeError(f"Erreur synthèse native: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du handler optimisé"""
        return {
            'handler_type': 'piper_native_optimized',
            'is_loaded': self.is_loaded,
            'load_time_ms': self.load_time_ms,
            'synthesis_count': self.synthesis_count,
            'target_latency_ms': self.target_latency_ms,
            'model_path': str(self.model_path),
            'device': self.device,
            'speaker_id': self.speaker_id
        }
    
    def cleanup(self):
        """Nettoyage des ressources"""
        if self.voice:
            # Libération du modèle si la méthode existe
            if hasattr(self.voice, 'cleanup'):
                self.voice.cleanup()
            elif hasattr(self.voice, 'close'):
                self.voice.close()
            
            self.voice = None
            self.is_loaded = False
            
        logging.info("PiperNativeOptimizedHandler nettoyé")
    
    def __del__(self):
        """Destructeur pour nettoyage automatique"""
        self.cleanup()


# =============================================================================
# FALLBACK HANDLER SI BINDING PYTHON NON DISPONIBLE
# =============================================================================

class PiperNativeFallbackHandler:
    """
    Handler de fallback si le binding Python Piper n'est pas disponible
    Utilise l'implémentation CLI existante avec optimisations mineures
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logging.warning(
            "⚠️ Binding Python Piper non disponible - Utilisation du fallback CLI"
        )
        
        # Import du handler CLI existant
        from ..tts_manager import PiperNativeHandler
        self._cli_handler = PiperNativeHandler(config)
    
    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """Délégation vers le handler CLI existant"""
        return await self._cli_handler.synthesize(text, voice, speed)
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du fallback"""
        stats = self._cli_handler.get_stats() if hasattr(self._cli_handler, 'get_stats') else {}
        stats['handler_type'] = 'piper_native_fallback'
        return stats
    
    def cleanup(self):
        """Nettoyage du fallback"""
        if hasattr(self._cli_handler, 'cleanup'):
            self._cli_handler.cleanup()


# =============================================================================
# FACTORY POUR SÉLECTION AUTOMATIQUE
# =============================================================================

def create_piper_native_handler(config: Dict[str, Any]):
    """
    Factory pour créer le meilleur handler Piper natif disponible
    
    Returns:
        PiperNativeOptimizedHandler si binding Python disponible
        PiperNativeFallbackHandler sinon
    """
    if PIPER_AVAILABLE:
        return PiperNativeOptimizedHandler(config)
    else:
        return PiperNativeFallbackHandler(config) 