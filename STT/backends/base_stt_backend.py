#!/usr/bin/env python3
"""
Interface de base pour tous les backends STT - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import time

# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

@dataclass
class STTResult:
    """Résultat de transcription STT standardisé"""
    text: str
    confidence: float
    segments: List[dict]
    processing_time: float
    device: str
    rtf: float  # Real-Time Factor
    backend_used: str
    success: bool
    error: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

def validate_rtx3090_mandatory():
    """Validation RTX 3090 selon standards SuperWhisper V6"""
    try:
        import torch
    except ImportError:
        raise RuntimeError("🚫 PyTorch non installé - RTX 3090 requise pour STT")
    
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise pour STT")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 3090" not in gpu_name:
        raise RuntimeError(f"🚫 GPU: {gpu_name} - RTX 3090 requise")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 VRAM {gpu_memory:.1f}GB insuffisante - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée pour STT: {gpu_name} ({gpu_memory:.1f}GB)")

class BaseSTTBackend(ABC):
    """Interface de base pour tous les backends STT"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le backend STT
        
        Args:
            config: Configuration du backend
        """
        validate_rtx3090_mandatory()
        
        self.config = config
        self.device = config.get('device', 'cuda:0')  # RTX 3090 après mapping
        self.model_name = config.get('model', 'unknown')
        
        # Métriques
        self.total_requests = 0
        self.total_errors = 0
        self.total_processing_time = 0.0
        
        print(f"🎤 Backend STT initialisé: {self.__class__.__name__}")
    
    @abstractmethod
    async def transcribe(self, audio: np.ndarray) -> STTResult:
        """
        Transcrit l'audio en texte
        
        Args:
            audio: Audio 16kHz mono float32
            
        Returns:
            STTResult avec transcription et métriques
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Vérifie l'état de santé du backend
        
        Returns:
            True si le backend est opérationnel
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du backend"""
        avg_processing_time = (
            self.total_processing_time / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        return {
            "backend_name": self.__class__.__name__,
            "model_name": self.model_name,
            "device": self.device,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_requests, 1),
            "avg_processing_time": avg_processing_time,
            "success_rate": (self.total_requests - self.total_errors) / max(self.total_requests, 1)
        }
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Surveillance mémoire GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                return {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total,
                    "free_gb": total - reserved,
                    "usage_percent": (reserved / total) * 100
                }
        except Exception:
            pass
        
        return {}
    
    def _record_request(self, processing_time: float, success: bool):
        """Enregistre les métriques d'une requête"""
        self.total_requests += 1
        self.total_processing_time += processing_time
        
        if not success:
            self.total_errors += 1 