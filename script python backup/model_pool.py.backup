#!/usr/bin/env python3
"""
ModelPool pour SuperWhisper V6
Charge et gère les modèles Whisper pour éviter les duplications en VRAM.
"""

import os
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