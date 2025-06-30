#!/usr/bin/env python3
"""
Utilitaires de traitement audio - SuperWhisper V6 STT
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Traitement et validation audio pour STT

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

import numpy as np
import logging
from typing import Tuple, Optional
import hashlib

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Processeur audio optimisé pour STT SuperWhisper V6"""
    
    TARGET_SAMPLE_RATE = 16000  # Whisper standard
    TARGET_CHANNELS = 1  # Mono
    
    @staticmethod
    def validate_audio_format(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bool:
        """
        Valide le format audio pour STT
        
        Args:
            audio: Array audio numpy
            sample_rate: Taux d'échantillonnage
            
        Returns:
            True si format valide
        """
        try:
            # Vérifications de base
            if not isinstance(audio, np.ndarray):
                logger.error("Audio doit être un numpy array")
                return False
            
            if audio.dtype != np.float32:
                logger.error(f"Audio dtype {audio.dtype} invalide - doit être float32")
                return False
            
            if len(audio.shape) > 1:
                logger.error(f"Audio shape {audio.shape} invalide - doit être mono (1D)")
                return False
            
            if sample_rate != AudioProcessor.TARGET_SAMPLE_RATE:
                logger.error(f"Sample rate {sample_rate} invalide - doit être {AudioProcessor.TARGET_SAMPLE_RATE}Hz")
                return False
            
            # Vérification durée
            duration = len(audio) / sample_rate
            if duration <= 0:
                logger.error("Durée audio invalide")
                return False
            
            if duration > 30.0:  # Max 30s
                logger.warning(f"Audio long: {duration:.1f}s > 30s")
            
            # Vérification amplitude
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 1.0:
                logger.error(f"Amplitude audio {max_amplitude} > 1.0")
                return False
            
            if max_amplitude < 0.001:  # Très faible signal
                logger.warning(f"Signal audio très faible: {max_amplitude}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation audio: {e}")
            return False
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """
        Normalise l'audio pour STT
        
        Args:
            audio: Array audio numpy
            
        Returns:
            Audio normalisé
        """
        try:
            # Copie pour éviter modification originale
            normalized = audio.copy()
            
            # Normalisation amplitude
            max_amplitude = np.max(np.abs(normalized))
            if max_amplitude > 0:
                normalized = normalized / max_amplitude * 0.95  # Éviter clipping
            
            # Suppression DC offset
            normalized = normalized - np.mean(normalized)
            
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erreur normalisation audio: {e}")
            return audio
    
    @staticmethod
    def compute_audio_hash(audio: np.ndarray) -> str:
        """
        Calcule hash audio pour cache
        
        Args:
            audio: Array audio numpy
            
        Returns:
            Hash MD5 de l'audio
        """
        try:
            # Convertir en bytes pour hash
            audio_bytes = audio.tobytes()
            return hashlib.md5(audio_bytes).hexdigest()
            
        except Exception as e:
            logger.error(f"Erreur calcul hash audio: {e}")
            return ""
    
    @staticmethod
    def detect_silence(audio: np.ndarray, threshold: float = 0.01) -> Tuple[bool, float]:
        """
        Détecte si l'audio est principalement silencieux
        
        Args:
            audio: Array audio numpy
            threshold: Seuil de détection silence
            
        Returns:
            (is_silent, energy_level)
        """
        try:
            # Calcul énergie RMS
            rms_energy = np.sqrt(np.mean(audio ** 2))
            
            is_silent = rms_energy < threshold
            
            return is_silent, float(rms_energy)
            
        except Exception as e:
            logger.error(f"Erreur détection silence: {e}")
            return False, 0.0
    
    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Supprime le silence au début et à la fin
        
        Args:
            audio: Array audio numpy
            threshold: Seuil de détection silence
            
        Returns:
            Audio sans silence aux extrémités
        """
        try:
            # Détection zones non-silencieuses
            energy = np.abs(audio)
            non_silent = energy > threshold
            
            if not np.any(non_silent):
                # Tout est silence, retourner audio court
                return audio[:1000] if len(audio) > 1000 else audio
            
            # Indices début/fin non-silence
            start_idx = np.argmax(non_silent)
            end_idx = len(audio) - np.argmax(non_silent[::-1]) - 1
            
            return audio[start_idx:end_idx + 1]
            
        except Exception as e:
            logger.error(f"Erreur trim silence: {e}")
            return audio

def validate_audio_format(audio: np.ndarray, sample_rate: int = 16000) -> bool:
    """
    Fonction utilitaire pour validation audio
    
    Args:
        audio: Array audio numpy
        sample_rate: Taux d'échantillonnage
        
    Returns:
        True si format valide
    """
    return AudioProcessor.validate_audio_format(audio, sample_rate) 