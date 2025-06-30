#!/usr/bin/env python3
"""
SOLUTION MEMORY LEAK GPU V3 - SuperWhisper V6 [STABLE WINDOWS]
🚨 CONFIGURATION: RTX 3090 CUDA:1 - Version simplifiée sans blocages

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

# Configuration simplifiée pour Windows
LEAK_THRESHOLD_MB = float(os.environ.get('LUXA_LEAK_THRESHOLD_MB', '100'))
TIMEOUT_SECONDS = int(os.environ.get('LUXA_GPU_TIMEOUT_S', '60'))  # Réduit à 60s

print("🚀 SOLUTION MEMORY LEAK GPU V3 - STABLE WINDOWS")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"⚙️ Seuil Memory Leak: {LEAK_THRESHOLD_MB}MB")
print(f"⏰ Timeout GPU: {TIMEOUT_SECONDS}s")

class GPUMemoryManager:
    """Gestionnaire simplifié des fuites mémoire GPU RTX 3090 [V3 STABLE]"""
    
    def __init__(self):
        self.device = "cuda:0"  # RTX 3090 via CUDA_VISIBLE_DEVICES='1'
        self.initial_memory = None
        self.lock = threading.Lock()  # Un seul lock simple
        self._validate_gpu()
        self._initialize_baseline()
    
    def _validate_gpu(self):
        """Validation critique RTX 3090"""
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    
    def _initialize_baseline(self):
        """Initialiser baseline mémoire APRÈS cleanup initial"""
        with self.lock:
            self._force_cleanup_internal()
            self.initial_memory = torch.cuda.memory_allocated(0)
            print(f"📏 Baseline mémoire: {self.initial_memory / 1024**3:.3f}GB")
    
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
                'utilization_pct': (reserved / total) * 100,
                'fragmentation_gb': reserved - allocated
            }
    
    def _force_cleanup_internal(self):
        """Cleanup interne simplifié (pas de signal sur Windows)"""
        try:
            # 1. Vider cache PyTorch
            torch.cuda.empty_cache()
            
            # 2. Garbage collection Python
            gc.collect()
            
            # 3. Synchronisation GPU (pas de timeout sur Windows)
            torch.cuda.synchronize()
            
            # 4. Reset statistiques mémoire compatible
            torch.cuda.reset_max_memory_allocated()
            
            # Version compatible PyTorch
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            print("🧹 Cleanup GPU complet")
            
        except Exception as e:
            print(f"⚠️ Erreur cleanup GPU: {e}")
    
    def force_cleanup(self):
        """Cleanup forcé public"""
        with self.lock:
            self._force_cleanup_internal()
    
    @contextlib.contextmanager
    def gpu_context(self, test_name: str = "unknown"):
        """Context manager simplifié avec cleanup automatique"""
        print(f"🔄 Début test GPU: {test_name}")
        
        with self.lock:  # Un seul lock
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
                # Cleanup automatique
                self._force_cleanup_internal()
                
                # Statistiques après
                stats_after = self.get_memory_stats()
                duration = time.time() - start_time
                
                # Rapport test
                memory_diff = stats_after['allocated_gb'] - stats_before['allocated_gb']
                threshold_gb = LEAK_THRESHOLD_MB / 1024
                
                print(f"📊 Test {test_name} terminé ({duration:.2f}s)")
                print(f"   Avant: {stats_before['allocated_gb']:.3f}GB")
                print(f"   Après: {stats_after['allocated_gb']:.3f}GB")
                print(f"   Diff: {memory_diff:+.3f}GB (seuil: {threshold_gb:.3f}GB)")
                
                if abs(memory_diff) > threshold_gb:
                    print(f"🚨 MEMORY LEAK: {memory_diff:+.3f}GB")
                else:
                    print("✅ Pas de memory leak")

# Instance globale simplifiée
gpu_manager = GPUMemoryManager()

def gpu_test_cleanup(test_name: Optional[str] = None):
    """Décorateur simplifié pour cleanup automatique"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = test_name or func.__name__
            with gpu_manager.gpu_context(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_no_memory_leak():
    """Validation memory leak avec seuil harmonisé"""
    if gpu_manager.initial_memory is None:
        print("⚠️ Baseline non initialisée")
        return False
        
    initial = gpu_manager.initial_memory / 1024**3
    current = torch.cuda.memory_allocated(0) / 1024**3
    diff = current - initial
    threshold_gb = LEAK_THRESHOLD_MB / 1024
    
    print(f"🔍 VALIDATION MEMORY LEAK")
    print(f"   Baseline: {initial:.3f}GB")
    print(f"   Actuelle: {current:.3f}GB")
    print(f"   Diff: {diff:+.3f}GB (seuil: {threshold_gb:.3f}GB)")
    
    if abs(diff) > threshold_gb:
        print(f"❌ MEMORY LEAK: {diff:+.3f}GB")
        return False
    else:
        print("✅ Aucun memory leak")
        return True

def emergency_gpu_reset():
    """Reset GPU d'urgence simplifié"""
    print("🚨 RESET GPU D'URGENCE")
    try:
        gpu_manager.force_cleanup()
        gpu_manager._initialize_baseline()
        print("✅ Reset terminé")
    except Exception as e:
        print(f"❌ Échec reset: {e}")

# =============================================================================
# TESTS SIMPLIFIÉS
# =============================================================================

@gpu_test_cleanup("test_model_simple")
def test_simple_model():
    """Test modèle simplifié"""
    device = gpu_manager.device
    print(f"   🔧 Création tenseur sur {device}")
    
    # Test plus léger
    model = torch.randn(500, 500, device=device, dtype=torch.float32)
    result = torch.matmul(model, model.t())
    
    print(f"   ✅ Calcul terminé: {result.shape}")
    return result.cpu()

def run_simple_tests():
    """Tests simplifiés sans blocage"""
    print("🔄 TESTS SIMPLIFIÉS V3")
    
    for i in range(3):  # Seulement 3 tests
        print(f"\n--- Test {i+1}/3 ---")
        
        try:
            result = test_simple_model()
            print(f"   Résultat: {result.shape}")
            validate_no_memory_leak()
            
        except Exception as e:
            print(f"❌ Erreur test {i+1}: {e}")
            emergency_gpu_reset()
    
    print("\n🏆 TESTS TERMINÉS")
    
    # Rapport final
    stats = gpu_manager.get_memory_stats()
    print(f"\n📊 STATISTIQUES FINALES:")
    print(f"   Mémoire utilisée: {stats['allocated_gb']:.3f}GB")
    print(f"   Fragmentation: {stats['fragmentation_gb']:.3f}GB")
    
    if validate_no_memory_leak():
        print("✅ V3 VALIDÉE - PRÊTE POUR PARALLÉLISATION")
        return True
    else:
        print("❌ Memory leak détecté")
        return False

if __name__ == "__main__":
    print("🧪 TEST SOLUTION V3 STABLE")
    
    # Stats initiales
    print(f"\n📊 Stats initiales:")
    initial_stats = gpu_manager.get_memory_stats()
    for key, val in initial_stats.items():
        print(f"   {key}: {val:.3f}")
    
    # Tests
    success = run_simple_tests()
    
    if success:
        print("\n🎯 SOLUTION V3 STABLE VALIDÉE")
    else:
        print("\n⚠️ Tests partiels") 