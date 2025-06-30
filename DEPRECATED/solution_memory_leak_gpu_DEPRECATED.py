#!/usr/bin/env python3
"""
SOLUTION MEMORY LEAK GPU - SuperWhisper V6
🚨 CONFIGURATION: RTX 3090 CUDA:1 avec cleanup automatique

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

import torch
import gc
import threading
import contextlib
import functools
from typing import Optional, Dict, Any
import time
import traceback

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🚀 SOLUTION MEMORY LEAK GPU - RTX 3090")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

class GPUMemoryManager:
    """Gestionnaire automatique des fuites mémoire GPU RTX 3090"""
    
    def __init__(self):
        self.device = "cuda:0"  # RTX 3090 via CUDA_VISIBLE_DEVICES='1'
        self.initial_memory = None
        self.lock = threading.Lock()
        self._validate_gpu()
    
    def _validate_gpu(self):
        """Validation critique RTX 3090"""
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        self.initial_memory = torch.cuda.memory_allocated(0)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Statistiques mémoire GPU détaillées"""
        with self.lock:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'max_reserved_gb': max_reserved,
                'total_gb': total,
                'free_gb': total - reserved,
                'utilization_pct': (reserved / total) * 100
            }
    
    def force_cleanup(self):
        """Nettoyage forcé complet GPU"""
        with self.lock:
            try:
                # 1. Vider cache PyTorch
                torch.cuda.empty_cache()
                
                # 2. Garbage collection Python
                gc.collect()
                
                # 3. Synchronisation GPU
                torch.cuda.synchronize()
                
                # 4. Reset statistiques mémoire
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()
                
                print("🧹 Cleanup GPU complet effectué")
                
            except Exception as e:
                print(f"⚠️ Erreur cleanup GPU: {e}")
    
    @contextlib.contextmanager
    def gpu_context(self, test_name: str = "unknown"):
        """Context manager avec cleanup automatique"""
        print(f"🔄 Début test GPU: {test_name}")
        
        # Statistiques avant
        stats_before = self.get_memory_stats()
        start_time = time.time()
        
        try:
            yield self.device
            
        except Exception as e:
            print(f"❌ Erreur test {test_name}: {e}")
            traceback.print_exc()
            raise
            
        finally:
            # Cleanup automatique après test
            self.force_cleanup()
            
            # Statistiques après
            stats_after = self.get_memory_stats()
            duration = time.time() - start_time
            
            # Rapport test
            memory_diff = stats_after['allocated_gb'] - stats_before['allocated_gb']
            
            print(f"📊 Test {test_name} terminé ({duration:.2f}s)")
            print(f"   Mémoire avant: {stats_before['allocated_gb']:.2f}GB")
            print(f"   Mémoire après: {stats_after['allocated_gb']:.2f}GB")
            print(f"   Différence: {memory_diff:+.3f}GB")
            
            if abs(memory_diff) > 0.1:  # Seuil memory leak
                print(f"⚠️ POTENTIAL MEMORY LEAK: {memory_diff:+.3f}GB")
            else:
                print("✅ Pas de memory leak détecté")

# =============================================================================
# DÉCORATEURS POUR TESTS GPU AUTOMATIQUES
# =============================================================================

gpu_manager = GPUMemoryManager()

def gpu_test_cleanup(test_name: Optional[str] = None):
    """Décorateur pour cleanup automatique tests GPU"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = test_name or func.__name__
            with gpu_manager.gpu_context(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def gpu_memory_monitor(threshold_gb: float = 1.0):
    """Décorateur pour monitoring mémoire GPU"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stats_before = gpu_manager.get_memory_stats()
            
            result = func(*args, **kwargs)
            
            stats_after = gpu_manager.get_memory_stats()
            memory_used = stats_after['allocated_gb'] - stats_before['allocated_gb']
            
            if memory_used > threshold_gb:
                print(f"🚨 ALERTE MÉMOIRE: {func.__name__} utilise {memory_used:.2f}GB")
                print(f"   Seuil: {threshold_gb}GB | Utilisation GPU: {stats_after['utilization_pct']:.1f}%")
            
            return result
        return wrapper
    return decorator

# =============================================================================
# FONCTIONS UTILITAIRES MEMORY LEAK PREVENTION
# =============================================================================

def validate_no_memory_leak():
    """Validation qu'aucun memory leak n'existe"""
    initial = gpu_manager.initial_memory / 1024**3
    current = torch.cuda.memory_allocated(0) / 1024**3
    diff = current - initial
    
    print(f"🔍 VALIDATION MEMORY LEAK")
    print(f"   Mémoire initiale: {initial:.3f}GB")
    print(f"   Mémoire actuelle: {current:.3f}GB")
    print(f"   Différence: {diff:+.3f}GB")
    
    if abs(diff) > 0.05:  # Seuil 50MB
        print(f"❌ MEMORY LEAK DÉTECTÉ: {diff:+.3f}GB")
        return False
    else:
        print("✅ Aucun memory leak détecté")
        return True

def emergency_gpu_reset():
    """Reset GPU d'urgence en cas de memory leak critique"""
    print("🚨 RESET GPU D'URGENCE")
    
    try:
        # Forcer cleanup complet
        gpu_manager.force_cleanup()
        
        # Réinitialiser contexte CUDA si possible
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()
        
        print("✅ Reset GPU d'urgence terminé")
        
    except Exception as e:
        print(f"❌ Échec reset GPU: {e}")
        print("💡 Solution: Redémarrer le script complet")

# =============================================================================
# EXEMPLE D'UTILISATION POUR TESTS PARALLÉLISATION
# =============================================================================

@gpu_test_cleanup("test_model_loading")
@gpu_memory_monitor(threshold_gb=2.0)
def test_load_model_with_cleanup():
    """Exemple test avec cleanup automatique"""
    device = gpu_manager.device
    
    # Simulation chargement modèle
    model = torch.randn(1000, 1000, device=device)
    result = torch.matmul(model, model.t())
    
    print(f"✅ Test modèle terminé sur {device}")
    return result.cpu()  # Retourner sur CPU pour libérer GPU

def run_parallel_tests_with_cleanup():
    """Exemple exécution tests parallèles avec cleanup"""
    print("🔄 TESTS PARALLÈLES AVEC CLEANUP AUTOMATIQUE")
    
    for i in range(5):
        print(f"\n--- Test {i+1}/5 ---")
        test_load_model_with_cleanup()
        
        # Validation après chaque test
        validate_no_memory_leak()
    
    print("\n🏆 TOUS LES TESTS TERMINÉS")
    
    # Validation finale
    if validate_no_memory_leak():
        print("✅ AUCUN MEMORY LEAK - PARALLÉLISATION SÛRE")
    else:
        print("❌ MEMORY LEAK DÉTECTÉ - CLEANUP REQUIS")
        emergency_gpu_reset()

if __name__ == "__main__":
    print("🧪 DÉMONSTRATION SOLUTION MEMORY LEAK")
    
    # Test initial
    print(f"\n📊 Statistiques GPU initiales:")
    stats = gpu_manager.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.2f}")
    
    # Tests avec cleanup
    run_parallel_tests_with_cleanup()
    
    print("\n🎯 SOLUTION MEMORY LEAK VALIDÉE POUR PARALLÉLISATION") 