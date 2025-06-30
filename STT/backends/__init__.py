#!/usr/bin/env python3
"""
Module STT Backends - SuperWhisper V6 Phase 4
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Backends STT avec fallback intelligent :
1. PrismSTTBackend (Principal) - Prism_Whisper2 RTX 3090
2. WhisperDirectBackend (Fallback 1) - faster-whisper direct
3. WhisperCPUBackend (Fallback 2) - CPU whisper
4. OfflineSTTBackend (Urgence) - Windows Speech API

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

from .base_stt_backend import BaseSTTBackend
from .prism_stt_backend import PrismSTTBackend

__all__ = [
    'BaseSTTBackend',
    'PrismSTTBackend'
] 