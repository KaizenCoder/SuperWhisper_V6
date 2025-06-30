#!/usr/bin/env python3
"""
ModelPool pour SuperWhisper V6
Charge et gère les modèles Whisper pour éviter les duplications en VRAM.

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

import time
import torch
from faster_whisper import WhisperModel
import logging
from typing import Dict, Optional

# Configuration GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

logger = logging.getLogger('ModelPool')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - ModelPool - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class ModelPool:
    """Charge et gère une instance unique de chaque modèle Whisper."""

    _instance = None
    _models: Dict[str, WhisperModel] = {}
    _lock = torch.multiprocessing.get_context("spawn").Lock() if torch.cuda.is_available() else None


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelPool, cls).__new__(cls)
            logger.info("Initializing ModelPool singleton.")
        return cls._instance

    def get_model(self, model_size: str, compute_type: str = "float16") -> Optional[WhisperModel]:
        """
        Récupère un modèle depuis le pool. Le charge si nécessaire.
        Cette méthode est thread-safe.
        """
        if self._lock is None:
            logger.error("CUDA not available, cannot use locks.")
            return None
        with self._lock:
            if model_size not in self._models:
                logger.info(f"Model '{model_size}' not in pool. Loading...")
                try:
                    start_time = time.time()
                    model = WhisperModel(
                        model_size,
                        device="cuda",
                        compute_type=compute_type
                    )
                    duration = time.time() - start_time
                    self._models[model_size] = model
                    logger.info(f"✅ Loaded model '{model_size}' in {duration:.2f}s.")
                    
                    # Vérification mémoire
                    mem_used_gb = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"   Current VRAM used: {mem_used_gb:.2f} GB")

                except Exception as e:
                    logger.error(f"❌ Failed to load model '{model_size}': {e}")
                    return None
            
            return self._models.get(model_size)

    def list_loaded_models(self) -> list[str]:
        """Liste les modèles actuellement chargés."""
        return list(self._models.keys())

# Singleton instance
model_pool = ModelPool() 