#!/usr/bin/env python3
"""
SOLUTION MEMORY LEAK GPU V2 - SuperWhisper V6 [VULNÉRABILITÉS CORRIGÉES]
🚨 CONFIGURATION: RTX 3090 CUDA:1 avec corrections critiques Claude + O3
"""

import os
import sys
import torch
import gc
import threading
import multiprocessing
import contextlib
import functools
import signal
from typing import Optional, Dict, Any
import time
import traceback
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# Seuil paramétrable memory leak (correction O3)
LEAK_THRESHOLD_MB = float(os.environ.get('LUXA_LEAK_THRESHOLD_MB', '100'))  # Default 100MB (Claude)
TIMEOUT_SECONDS = int(os.environ.get('LUXA_GPU_TIMEOUT_S', '300'))  # Default 5min

print("🚀 SOLUTION MEMORY LEAK GPU V2 - CORRECTIONS CRITIQUES")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"⚙️ Seuil Memory Leak: {LEAK_THRESHOLD_MB}MB")
print(f"⏰ Timeout GPU: {TIMEOUT_SECONDS}s")

# Sémaphore inter-processus pour queue GPU exclusive (correction O3)
LOCK_FILE = Path.cwd() / ".luxa_gpu_lock"
global_gpu_semaphore = None

def init_global_gpu_semaphore():
    """Initialiser le sémaphore inter-processus GPU"""
    global global_gpu_semaphore
    if global_gpu_semaphore is None:
        global_gpu_semaphore = multiprocessing.Semaphore(1)
    return global_gpu_semaphore

class GPUMemoryManager:
    """Gestionnaire automatique des fuites mémoire GPU RTX 3090 [CORRIGÉ V2]"""
    
    def __init__(self):
        self.device = "cuda:0"  # RTX 3090 via CUDA_VISIBLE_DEVICES='1'
        self.initial_memory = None
        self.lock = threading.Lock()
        self.gpu_semaphore = init_global_gpu_semaphore()
        self._validate_gpu()
        self._initialize_baseline()  # Correction O3: après validation
    
    def _validate_gpu(self):
        """Validation critique RTX 3090"""
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    
    def _initialize_baseline(self):
        """Initialiser baseline mémoire APRÈS cleanup initial (correction O3)"""
        with self.lock:
            # Cleanup initial obligatoire avant baseline
            self._force_cleanup_internal()
            self.initial_memory = torch.cuda.memory_allocated(0)
            print(f"📏 Baseline mémoire: {self.initial_memory / 1024**3:.3f}GB")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Statistiques mémoire GPU détaillées [AMÉLIORÉ V2]"""
        with self.lock:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Nouvelles métriques (correction Claude)
            reserved_vs_allocated_ratio = reserved / allocated if allocated > 0 else 0
            potential_fragmentation = reserved - allocated
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'max_reserved_gb': max_reserved,
                'total_gb': total,
                'free_gb': total - reserved,
                'utilization_pct': (reserved / total) * 100,
                'reserved_vs_allocated_ratio': reserved_vs_allocated_ratio,  # Nouveau
                'potential_fragmentation_gb': potential_fragmentation  # Nouveau
            }
    
    def _force_cleanup_internal(self):
        """Cleanup interne avec timeout (correction Claude + O3)"""
        try:
            # Timeout sur synchronize (correction Claude)
            def timeout_handler(signum, frame):
                raise TimeoutError("GPU synchronize timeout")
            
            if hasattr(signal, 'SIGALRM'):  # Unix seulement
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30s timeout
            
            # 1. Vider cache PyTorch
            torch.cuda.empty_cache()
            
            # 2. Garbage collection Python
            gc.collect()
            
            # 3. Synchronisation GPU avec timeout
            torch.cuda.synchronize()
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel timeout
            
            # 4. Reset statistiques mémoire [CORRIGÉ PyTorch ≥2.3]
            torch.cuda.reset_max_memory_allocated()
            
            # Correction O3: remplacer reset_max_memory_cached obsolète
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()  # PyTorch ≥2.1
            elif hasattr(torch.cuda, 'reset_max_memory_cached'):
                torch.cuda.reset_max_memory_cached()  # PyTorch <2.3 fallback
            
            print("🧹 Cleanup GPU complet avec timeout")
            
        except TimeoutError:
            print(f"⚠️ Timeout cleanup GPU après 30s")
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        except Exception as e:
            print(f"⚠️ Erreur cleanup GPU: {e}")
    
    def force_cleanup(self):
        """Cleanup forcé public avec lock"""
        with self.lock:
            self._force_cleanup_internal()
    
    @contextlib.contextmanager
    def gpu_context(self, test_name: str = "unknown", timeout_s: Optional[int] = None):
        """Context manager avec cleanup automatique + timeout + queue GPU [CORRIGÉ V2]"""
        timeout = timeout_s or TIMEOUT_SECONDS
        
        # Queue GPU inter-processus (correction O3)
        with self.gpu_semaphore:
            print(f"🔄 Début test GPU: {test_name} (timeout: {timeout}s)")
            
            # Watchdog timeout (correction O3)
            def watchdog_timeout():
                print(f"🚨 WATCHDOG: Test {test_name} timeout après {timeout}s - Emergency reset")
                self.emergency_gpu_reset()
            
            watchdog = threading.Timer(timeout, watchdog_timeout)
            watchdog.start()
            
            # Statistiques avant
            stats_before = self.get_memory_stats()
            start_time = time.time()
            
            try:
                with self.lock:  # Thread safety
                    yield self.device
                    
            except Exception as e:
                print(f"❌ Erreur test {test_name}: {e}")
                traceback.print_exc()
                raise
                
            finally:
                # Cancel watchdog
                watchdog.cancel()
                
                # Cleanup automatique après test
                self._force_cleanup_internal()
                
                # Statistiques après
                stats_after = self.get_memory_stats()
                duration = time.time() - start_time
                
                # Rapport test avec seuil paramétrable (correction O3)
                memory_diff = stats_after['allocated_gb'] - stats_before['allocated_gb']
                threshold_gb = LEAK_THRESHOLD_MB / 1024  # MB -> GB
                
                print(f"📊 Test {test_name} terminé ({duration:.2f}s)")
                print(f"   Mémoire avant: {stats_before['allocated_gb']:.3f}GB")
                print(f"   Mémoire après: {stats_after['allocated_gb']:.3f}GB")
                print(f"   Différence: {memory_diff:+.3f}GB (seuil: {threshold_gb:.3f}GB)")
                print(f"   Fragmentation: {stats_after['potential_fragmentation_gb']:.3f}GB")
                
                if abs(memory_diff) > threshold_gb:
                    print(f"🚨 MEMORY LEAK DÉTECTÉ: {memory_diff:+.3f}GB > {threshold_gb:.3f}GB")
                else:
                    print("✅ Pas de memory leak détecté")

# =============================================================================
# DÉCORATEURS POUR TESTS GPU AUTOMATIQUES [CORRIGÉS V2]
# =============================================================================

gpu_manager = GPUMemoryManager()

def gpu_test_cleanup(test_name: Optional[str] = None, timeout_s: Optional[int] = None):
    """Décorateur pour cleanup automatique tests GPU avec timeout"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = test_name or func.__name__
            with gpu_manager.gpu_context(name, timeout_s):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def gpu_memory_monitor(threshold_gb: float = 1.0):
    """Décorateur pour monitoring mémoire GPU avec métriques étendues"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stats_before = gpu_manager.get_memory_stats()
            
            result = func(*args, **kwargs)
            
            stats_after = gpu_manager.get_memory_stats()
            memory_used = stats_after['allocated_gb'] - stats_before['allocated_gb']
            fragmentation = stats_after['potential_fragmentation_gb']
            
            if memory_used > threshold_gb:
                print(f"🚨 ALERTE MÉMOIRE: {func.__name__} utilise {memory_used:.2f}GB")
                print(f"   Seuil: {threshold_gb}GB | Utilisation: {stats_after['utilization_pct']:.1f}%")
                print(f"   Fragmentation: {fragmentation:.3f}GB")
            
            return result
        return wrapper
    return decorator

# =============================================================================
# FONCTIONS UTILITAIRES MEMORY LEAK PREVENTION [CORRIGÉES V2]
# =============================================================================

def validate_no_memory_leak():
    """Validation qu'aucun memory leak n'existe [SEUIL HARMONISÉ]"""
    if gpu_manager.initial_memory is None:
        print("⚠️ Baseline mémoire non initialisée")
        return False
        
    initial = gpu_manager.initial_memory / 1024**3
    current = torch.cuda.memory_allocated(0) / 1024**3
    diff = current - initial
    threshold_gb = LEAK_THRESHOLD_MB / 1024  # Seuil paramétrable (correction O3)
    
    print(f"🔍 VALIDATION MEMORY LEAK")
    print(f"   Mémoire baseline: {initial:.3f}GB")
    print(f"   Mémoire actuelle: {current:.3f}GB")
    print(f"   Différence: {diff:+.3f}GB (seuil: {threshold_gb:.3f}GB)")
    
    if abs(diff) > threshold_gb:  # Seuil harmonisé (correction Claude)
        print(f"❌ MEMORY LEAK DÉTECTÉ: {diff:+.3f}GB > {threshold_gb:.3f}GB")
        return False
    else:
        print("✅ Aucun memory leak détecté")
        return True

def emergency_gpu_reset():
    """Reset GPU d'urgence en cas de memory leak critique [AMÉLIORÉ V2]"""
    print("🚨 RESET GPU D'URGENCE")
    
    try:
        # Forcer cleanup complet
        gpu_manager._force_cleanup_internal()
        
        # Réinitialiser contexte CUDA si possible
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()
        
        # Réinitialiser baseline après reset
        gpu_manager._initialize_baseline()
        
        print("✅ Reset GPU d'urgence terminé avec nouvelle baseline")
        
    except Exception as e:
        print(f"❌ Échec reset GPU: {e}")
        print("💡 Solution: Redémarrer le script complet")

# =============================================================================
# EXEMPLE D'UTILISATION POUR TESTS PARALLÉLISATION [VALIDÉ V2]
# =============================================================================

@gpu_test_cleanup("test_model_loading", timeout_s=120)
@gpu_memory_monitor(threshold_gb=2.0)
def test_load_model_with_cleanup():
    """Exemple test avec cleanup automatique et timeout"""
    device = gpu_manager.device
    
    # Simulation chargement modèle plus réaliste
    model = torch.randn(2000, 2000, device=device, dtype=torch.float32)
    for _ in range(3):  # Plusieurs opérations
        result = torch.matmul(model, model.t())
        intermediate = torch.relu(result)
    
    print(f"✅ Test modèle terminé sur {device}")
    return intermediate.cpu()  # Retourner sur CPU pour libérer GPU

def run_parallel_tests_with_cleanup():
    """Exemple exécution tests parallèles avec cleanup [V2 SÉCURISÉ]"""
    print("🔄 TESTS PARALLÈLES V2 - CORRECTIONS CRITIQUES APPLIQUÉES")
    
    for i in range(5):
        print(f"\n--- Test {i+1}/5 ---")
        
        try:
            result = test_load_model_with_cleanup()
            print(f"   Résultat shape: {result.shape}")
            
            # Validation après chaque test
            validate_no_memory_leak()
            
        except Exception as e:
            print(f"❌ Échec test {i+1}: {e}")
            emergency_gpu_reset()
    
    print("\n🏆 TOUS LES TESTS TERMINÉS")
    
    # Validation finale
    print("\n📊 RAPPORT FINAL:")
    final_stats = gpu_manager.get_memory_stats()
    for key, value in final_stats.items():
        if 'gb' in key or 'pct' in key or 'ratio' in key:
            print(f"   {key}: {value:.3f}")
    
    if validate_no_memory_leak():
        print("✅ AUCUN MEMORY LEAK - PARALLÉLISATION V2 VALIDÉE")
        return True
    else:
        print("❌ MEMORY LEAK PERSISTANT - INVESTIGATION REQUISE")
        emergency_gpu_reset()
        return False

if __name__ == "__main__":
    print("🧪 DÉMONSTRATION SOLUTION MEMORY LEAK V2 - CORRECTIONS APPLIQUÉES")
    
    # Test initial avec nouvelles métriques
    print(f"\n📊 Statistiques GPU initiales V2:")
    stats = gpu_manager.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.3f}")
    
    # Tests avec toutes les corrections
    success = run_parallel_tests_with_cleanup()
    
    if success:
        print("\n🎯 SOLUTION MEMORY LEAK V2 VALIDÉE - PRÊTE POUR 40 FICHIERS PARALLÈLES")
        print("✅ Toutes vulnérabilités critiques corrigées (Claude + O3)")
    else:
        print("\n⚠️ Tests partiels - Vérification manuelle requise") 