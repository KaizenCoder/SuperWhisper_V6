#!/usr/bin/env python3
"""
Utils Audio pour TTS SuperWhisper V6
====================================
• Utilitaires conversion audio PCM → WAV
• Validation format WAV
• Optimisé pour RTX 3090

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

import io
import wave
import struct
from typing import Tuple, Optional
import numpy as np

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 22050, channels: int = 1, sampwidth: int = 2) -> bytes:
    """
    Convertit des données PCM brutes en format WAV
    
    Args:
        pcm_data: Données PCM brutes
        sample_rate: Fréquence d'échantillonnage (Hz)
        channels: Nombre de canaux (1=mono, 2=stéréo)
        sampwidth: Largeur échantillon en bytes (2=16bit)
    
    Returns:
        bytes: Données WAV complètes avec header
    """
    try:
        # Créer un buffer WAV en mémoire
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        wav_buffer.seek(0)
        return wav_buffer.read()
        
    except Exception as e:
        raise RuntimeError(f"Erreur conversion PCM→WAV: {e}")

def is_valid_wav(data: bytes) -> bool:
    """
    Vérifie si les données sont un fichier WAV valide
    
    Args:
        data: Données à vérifier
    
    Returns:
        bool: True si WAV valide, False sinon
    """
    try:
        if len(data) < 44:  # Header WAV minimum = 44 bytes
            return False
        
        # Vérifier signature RIFF
        if data[:4] != b'RIFF':
            return False
        
        # Vérifier format WAVE
        if data[8:12] != b'WAVE':
            return False
        
        return True
        
    except Exception:
        return False

def get_wav_info(wav_data: bytes) -> Optional[Tuple[int, int, int]]:
    """
    Extrait les informations d'un fichier WAV
    
    Args:
        wav_data: Données WAV
    
    Returns:
        Tuple[sample_rate, channels, duration_ms] ou None si erreur
    """
    try:
        if not is_valid_wav(wav_data):
            return None
        
        wav_buffer = io.BytesIO(wav_data)
        
        with wave.open(wav_buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            frames = wav_file.getnframes()
            
            duration_ms = int((frames / sample_rate) * 1000)
            
            return sample_rate, channels, duration_ms
            
    except Exception:
        return None

def numpy_to_wav(audio_array: np.ndarray, sample_rate: int = 22050) -> bytes:
    """
    Convertit un array numpy en WAV
    
    Args:
        audio_array: Array numpy audio (float32 ou int16)
        sample_rate: Fréquence d'échantillonnage
    
    Returns:
        bytes: Données WAV
    """
    try:
        # Normaliser si nécessaire
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            # Convertir float [-1,1] vers int16
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * 32767).astype(np.int16)
        
        # Convertir en bytes
        pcm_data = audio_array.tobytes()
        
        # Convertir en WAV
        return pcm_to_wav(pcm_data, sample_rate, channels=1, sampwidth=2)
        
    except Exception as e:
        raise RuntimeError(f"Erreur conversion numpy→WAV: {e}")

def wav_to_numpy(wav_data: bytes) -> Tuple[np.ndarray, int]:
    """
    Convertit des données WAV en array numpy
    
    Args:
        wav_data: Données WAV
    
    Returns:
        Tuple[audio_array, sample_rate]
    """
    try:
        wav_buffer = io.BytesIO(wav_data)
        
        with wave.open(wav_buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            frames = wav_file.readframes(-1)
        
        # Convertir en numpy array
        if sampwidth == 2:  # 16-bit
            audio_array = np.frombuffer(frames, dtype=np.int16)
        elif sampwidth == 4:  # 32-bit
            audio_array = np.frombuffer(frames, dtype=np.int32)
        else:
            raise ValueError(f"Format non supporté: {sampwidth} bytes par échantillon")
        
        # Gérer multi-canaux
        if channels > 1:
            audio_array = audio_array.reshape(-1, channels)
            audio_array = audio_array.mean(axis=1)  # Convertir en mono
        
        # Normaliser vers float32 [-1, 1]
        if sampwidth == 2:
            audio_array = audio_array.astype(np.float32) / 32767.0
        elif sampwidth == 4:
            audio_array = audio_array.astype(np.float32) / 2147483647.0
        
        return audio_array, sample_rate
        
    except Exception as e:
        raise RuntimeError(f"Erreur conversion WAV→numpy: {e}") 