#!/usr/bin/env python3
"""
Module STT - SuperWhisper V6 Phase 4
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

# Configuration GPU obligatoire pour tout le module STT
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎤 Module STT - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Imports principaux
from .unified_stt_manager import UnifiedSTTManager
from .cache_manager import STTCache

__all__ = [
    'UnifiedSTTManager',
    'STTCache'
]

# Module STT SuperWhisper V6 