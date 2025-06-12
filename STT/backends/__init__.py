#!/usr/bin/env python3
"""
SuperWhisper V6 - Backends STT
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Backends STT avec fallback intelligent :
1. PrismSTTBackend (Principal) - Prism_Whisper2 RTX 3090
2. WhisperDirectBackend (Fallback 1) - faster-whisper direct
3. WhisperCPUBackend (Fallback 2) - CPU whisper
4. OfflineSTTBackend (Urgence) - Windows Speech API
"""

from .base_stt_backend import BaseSTTBackend
from .prism_stt_backend import PrismSTTBackend

__all__ = [
    'BaseSTTBackend',
    'PrismSTTBackend'
] 