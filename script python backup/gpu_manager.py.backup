#!/usr/bin/env python3
"""
GPU Manager - Luxa v1.1
========================
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Gestionnaire GPU dynamique avec détection automatique et mapping intelligent.
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 Ti (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Maintenant imports normaux...
import logging
import torch
import subprocess
from collections import defaultdict
from typing import Dict, Optional, Any

def validate_rtx3090_mandatory():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Singleton instance
_gpu_manager_instance = None

def is_pytest_running():
    """Vérifie si on est dans un contexte de test pytest."""
    return 'pytest' in sys.modules

class GPUManager:
    def __init__(self):
        """
        Initialise le gestionnaire de GPU avec RTX 3090 exclusive.
        Détecte si les tests sont en cours pour utiliser une configuration simulée.
        """
        # Validation RTX 3090 obligatoire (sauf en mode test)
        if not is_pytest_running():
            validate_rtx3090_mandatory()
        
        logger.debug("Initialisation du GPUManager RTX 3090...")
        self.gpu_map = {}
        self.device_names = {}
        self.device_capabilities: Dict[int, Dict[str, Any]] = {}

        if is_pytest_running():
            logger.warning("Mode test (pytest) détecté. Initialisation du GPUManager avec données simulées RTX 3090.")
            self.device_capabilities = {
                0: {'name': 'Mock RTX 3090', 'total_memory': 24_000_000_000}  # Seule GPU visible
            }
            self.is_mock = True
            # Build a mock gpu_map as well
            self.gpu_map = self._build_gpu_map()
        else:
            self.device_capabilities = self._analyze_devices()
            self.is_mock = False
            # Build the real gpu_map
            self.gpu_map = self._build_gpu_map()

    def _build_gpu_map(self) -> Dict[str, int]:
        """Construit un mapping RTX 3090 exclusif"""
        # Avec CUDA_VISIBLE_DEVICES='1', seule RTX 3090 est visible comme device 0
        return self._rtx3090_exclusive_mapping()
        
    def _rtx3090_exclusive_mapping(self) -> Dict[str, int]:
        """Mapping exclusif RTX 3090 - Toutes les tâches sur device 0 (RTX 3090 seule visible)"""
        if not torch.cuda.is_available():
            logger.error("❌ CUDA non disponible - RTX 3090 requise")
            return {}
            
        logger.info("🎯 Configuration RTX 3090 exclusive via CUDA_VISIBLE_DEVICES='1'")
        
        # RTX 3090 seule visible = device 0 pour tout
        gpu_map = {
            "llm": 0,      # RTX 3090 (seule visible)
            "stt": 0,      # RTX 3090 (seule visible)  
            "tts": 0,      # RTX 3090 (seule visible)
            "3090": 0,     # RTX 3090 (seule visible)
            "fallback": 0  # RTX 3090 (seule visible)
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logger.info(f"✅ RTX 3090 mappée: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        
        logger.info(f"🎯 GPU RTX 3090 mapping: {gpu_map}")
        return gpu_map
        
    def _analyze_devices(self) -> Dict[int, Dict[str, Any]]:
        """Analyse les capacités de la RTX 3090 (seule GPU visible)"""
        capabilities = {}
        
        if not torch.cuda.is_available():
            return capabilities
            
        # Avec CUDA_VISIBLE_DEVICES='1', seul device 0 existe (RTX 3090)
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        
        capabilities[0] = {
            "name": props.name,
            "total_memory_gb": total / 1024**3,
            "free_memory_gb": free / 1024**3,
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessor_count": props.multi_processor_count,
            "max_threads_per_block": getattr(props, 'max_threads_per_block', 1024)
        }
        
        logger.info(f"🔧 RTX 3090 analysée: {capabilities[0]['name']} ({capabilities[0]['total_memory_gb']:.1f}GB)")
        return capabilities
        
    def get_device(self, purpose: str = "llm") -> str:
        """Retourne le device RTX 3090 pour tout usage"""
        # RTX 3090 seule visible = cuda:0 pour tout
        if torch.cuda.is_available():
            return "cuda:0"  # RTX 3090 seule visible
        else:
            logger.warning("⚠️ CUDA non disponible - fallback CPU")
            return "cpu"
        
    def get_device_index(self, purpose: str = "llm") -> int:
        """Retourne l'index RTX 3090 (toujours 0 car seule visible)"""
        return 0  # RTX 3090 seule visible = index 0
        
    def can_load_model(self, model_size_gb: float, purpose: str = "llm") -> bool:
        """Vérifie si on peut charger un modèle sur RTX 3090"""
        if not torch.cuda.is_available():
            return False
            
        # RTX 3090 seule visible = device 0
        if 0 not in self.device_capabilities:
            return False
            
        free_gb = self.device_capabilities[0]["free_memory_gb"]
        
        # Marge de sécurité de 2GB
        can_load = free_gb > (model_size_gb + 2.0)
        
        logger.info(f"🔍 RTX 3090 ({purpose}): {free_gb:.1f}GB libre, "
                   f"modèle {model_size_gb:.1f}GB - {'✅' if can_load else '❌'}")
              
        return can_load
        
    def get_optimal_batch_size(self, purpose: str = "llm", base_batch_size: int = 1) -> int:
        """Calcule la taille de batch optimale selon la VRAM RTX 3090"""
        if not torch.cuda.is_available():
            return 1
            
        # RTX 3090 seule visible = device 0
        if 0 not in self.device_capabilities:
            return base_batch_size
            
        free_gb = self.device_capabilities[0]["free_memory_gb"]
        
        # Heuristique simple: 1 batch par 4GB disponible
        optimal_batch = max(1, int(free_gb / 4) * base_batch_size)
        optimal_batch = min(optimal_batch, 16)  # Cap à 16
        
        logger.info(f"📊 Batch size optimal RTX 3090 pour {purpose}: {optimal_batch}")
        return optimal_batch
        
    def update_memory_info(self):
        """Met à jour les informations mémoire RTX 3090"""
        if not torch.cuda.is_available():
            return
            
        # RTX 3090 seule visible = device 0
        free, total = torch.cuda.mem_get_info(0)
        
        if 0 in self.device_capabilities:
            self.device_capabilities[0]["free_memory_gb"] = free / 1024**3
            self.device_capabilities[0]["total_memory_gb"] = total / 1024**3
        
    def print_status(self):
        """Affiche le statut RTX 3090"""
        logger.info("🖥️ GPU MANAGER STATUS RTX 3090")
        logger.info("="*50)
        
        if not torch.cuda.is_available():
            logger.info("❌ CUDA non disponible")
            return
            
        logger.info(f"🎮 Configuration: RTX 3090 exclusive via CUDA_VISIBLE_DEVICES='1'")
        logger.info(f"📊 GPU mapping: {self.gpu_map}")
        
        for idx, caps in self.device_capabilities.items():
            logger.info(f"GPU {idx}: {caps['name']}")
            logger.info(f"   💾 VRAM: {caps['free_memory_gb']:.1f}/{caps['total_memory_gb']:.1f} GB")
            logger.info(f"   🔧 Compute: {caps.get('compute_capability', 'N/A')}")
            
    def get_best_device_for_model(self, model_size_gb: float) -> tuple[str, int]:
        """Retourne le meilleur device pour un modèle (RTX 3090 exclusive)"""
        if not torch.cuda.is_available():
            return "cpu", -1
            
        # RTX 3090 seule visible = device 0
        if self.can_load_model(model_size_gb, "llm"):
            return "cuda:0", 0  # RTX 3090
        else:
            logger.warning(f"⚠️ Modèle {model_size_gb:.1f}GB trop volumineux pour RTX 3090")
            return "cpu", -1

def get_gpu_manager() -> GPUManager:
    """Retourne l'instance singleton du GPU Manager RTX 3090"""
    global _gpu_manager_instance
    if _gpu_manager_instance is None:
        _gpu_manager_instance = GPUManager()
    return _gpu_manager_instance

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    
    # Test du GPU Manager
    gm = get_gpu_manager()
    gm.print_status()
    
    # Tests
    logger.info("\n🧪 Tests RTX 3090:")
    logger.info(f"Device LLM: {gm.get_device('llm')}")
    logger.info(f"Device STT: {gm.get_device('stt')}")
    logger.info(f"Device TTS: {gm.get_device('tts')}")
    
    # Test chargement modèle
    can_load = gm.can_load_model(7.0, "llm")  # 7GB LLM
    logger.info(f"Peut charger LLM 7GB: {can_load}")
    
    # Batch size optimal
    batch = gm.get_optimal_batch_size("stt", 2)
    logger.info(f"Batch optimal STT: {batch}")
    
    logger.info("✅ Tests GPU Manager RTX 3090 terminés") 