#!/usr/bin/env python3
"""
SuperWhisper V6 - Module STT (Speech-to-Text)
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Module principal pour la transcription vocale avec Prism_Whisper2
Architecture inspirée du succès TTS Phase 3 (29.5ms latence)
"""

__version__ = "6.4.0"
__author__ = "SuperWhisper V6 Team"

# Configuration GPU RTX 3090 obligatoire
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 SuperWhisper V6 STT - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

from .unified_stt_manager import UnifiedSTTManager
from .backends.base_stt_backend import BaseSTTBackend

__all__ = [
    'UnifiedSTTManager',
    'BaseSTTBackend'
] 