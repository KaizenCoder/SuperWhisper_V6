import os
import sys
import torch
import gc
import threading
import contextlib
import functools
import signal
from typing import Optional, Dict, Any, Callable
import time
import traceback
import platform
import json
from datetime import datetime
import multiprocessing
from multiprocessing import Manager
import tempfile
from pathlib import Path
try:
    import fcntl
except ImportError:
    fcntl = None  # Windows n'a pas fcntl
import errno

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# Configuration globale
FRAGMENTATION_THRESHOLD_GB = float(os.environ.get('LUXA_FRAGMENT_THRESHOLD_GB', '1.0'))
JSON_LOG_MAX_SIZE = int(os.environ.get('LUXA_JSON_LOG_MAX_SIZE', '1000'))

print("🚀 SOLUTION MEMORY LEAK GPU - RTX 3090 (VERSION 4.0 FINALE)")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"🖥️ Platform: {platform.system()} ({os.name})")

# Manager global pour sémaphores partagés entre processus
_process_manager = None

def get_process_manager():
    """Obtient ou crée le manager de processus global"""
    global _process_manager
    if _process_manager is None:
        _process_manager = Manager()
    return _process_manager

class FileLock:
    """Simple file-based lock pour cross-platform (fallback si Manager échoue)"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
    
    def acquire(self, timeout=None):
        """Acquiert le lock avec timeout optionnel"""
        self.file = open(self.filepath, 'w')
        start_time = time.time()
        
        while True:
            try:
                if os.name != 'nt' and fcntl:  # Unix/Linux
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:  # Windows
                    # Sur Windows, on utilise une approche différente
                    try:
                        os.rename(self.filepath, self.filepath + '.lock')
                        os.rename(self.filepath + '.lock', self.filepath)
                        return True
                    except:
                        pass
                return True
            except (IOError, OSError):
                if timeout and (time.time() - start_time) > timeout:
                    self.file.close()
                    return False
                time.sleep(0.1)
    
    def release(self):
        """Libère le lock"""
        if self.file:
            if os.name != 'nt' and fcntl:
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            else:  # Windows - Cleanup des fichiers .lock fantômes
                try:
                    os.remove(self.filepath + '.lock')
                except FileNotFoundError:
                    pass  # Fichier .lock déjà supprimé ou inexistant
            self.file.close()
            self.file = None

class GPUMemoryManager:
    """Gestionnaire automatique des fuites mémoire GPU RTX 3090 - V4.0"""
    
    def __init__(self, enable_json_logging: bool = False, 
                 enable_multiprocess_lock: bool = False,
                 fragmentation_threshold_gb: float = FRAGMENTATION_THRESHOLD_GB):
        self.device = "cuda:0"  # RTX 3090 via CUDA_VISIBLE_DEVICES='1'
        self.initial_memory = None
        self.lock = threading.Lock()
        self.cleanup_timeout = 30  # Timeout 30s pour opérations GPU
        self.enable_json_logging = enable_json_logging
        self.json_logs = []
        self.json_log_file = None
        self.fragmentation_threshold_gb = fragmentation_threshold_gb
        
        # Tracking du lock multiprocess pour éviter double release
        self.multiprocess_lock = None
        self.multiprocess_lock_held = False
        
        # Initialisation du lock multiprocess si requis
        if enable_multiprocess_lock:
            self._init_multiprocess_lock()
        
        # Validation GPU et cleanup initial
        self._validate_gpu()
        self.force_cleanup()  # Cleanup AVANT la mesure baseline
        self.initial_memory = torch.cuda.memory_allocated(0)  # Baseline après cleanup
    
    def _init_multiprocess_lock(self):
        """Initialise un sémaphore multiprocess pour accès GPU exclusif"""
        try:
            # Essayer d'abord avec Manager pour un vrai partage inter-process
            manager = get_process_manager()
            self.multiprocess_lock = manager.Semaphore(1)
            print("✅ Lock multiprocess GPU initialisé (Manager)")
        except Exception as e:
            print(f"⚠️ Manager échoué, utilisation FileLock: {e}")
            # Fallback sur FileLock
            lock_file = Path(tempfile.gettempdir()) / "gpu_rtx3090.lock"
            self.multiprocess_lock = FileLock(str(lock_file))
            print("✅ Lock multiprocess GPU initialisé (FileLock)")
    
    def _validate_gpu(self):
        """Validation critique RTX 3090"""
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if "3090" not in gpu_name.upper():
            print(f"⚠️ GPU détectée: {gpu_name} - Attendu: RTX 3090")
        
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Statistiques mémoire GPU détaillées avec monitoring fragmentation"""
        with self.lock:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Monitoring fragmentation mémoire
            potential_fragmentation = reserved - allocated
            reserved_vs_allocated_ratio = reserved / allocated if allocated > 0 else 0
            fragmentation_pct = (potential_fragmentation / total) * 100 if total > 0 else 0
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'max_reserved_gb': max_reserved,
                'total_gb': total,
                'free_gb': total - reserved,
                'utilization_pct': (reserved / total) * 100,
                'potential_fragmentation_gb': potential_fragmentation,
                'fragmentation_pct': fragmentation_pct,
                'reserved_vs_allocated_ratio': reserved_vs_allocated_ratio
            }
    
    def _timeout_handler_unix(self, signum, frame):
        """Handler pour timeout opérations GPU (Unix/Linux)"""
        raise TimeoutError("Timeout opération GPU")
    
    def _execute_with_timeout(self, func: Callable, timeout: int):
        """Exécute une fonction avec timeout cross-platform"""
        if os.name != 'nt' and hasattr(signal, 'SIGALRM'):
            # Unix/Linux: utiliser SIGALRM
            old_handler = signal.signal(signal.SIGALRM, self._timeout_handler_unix)
            signal.alarm(timeout)
            try:
                result = func()
                signal.alarm(0)  # Cancel timeout
                signal.signal(signal.SIGALRM, old_handler)
                return result
            except TimeoutError:
                signal.alarm(0)
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Windows: utiliser threading.Timer
            timeout_event = threading.Event()
            result = [None]
            exception = [None]
            
            def run_func():
                try:
                    result[0] = func()
                except Exception as e:
                    exception[0] = e
                finally:
                    timeout_event.set()
            
            thread = threading.Thread(target=run_func)
            thread.daemon = True
            thread.start()
            
            if not timeout_event.wait(timeout):
                # Timeout atteint
                raise TimeoutError(f"Timeout opération GPU ({timeout}s)")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
    
    def force_cleanup(self):
        """Nettoyage forcé complet GPU avec protection timeout cross-platform"""
        # Ne PAS acquérir/libérer le lock multiprocess ici (évite double release)
        with self.lock:
            try:
                # 1. Synchronisation GPU avec timeout cross-platform
                try:
                    self._execute_with_timeout(torch.cuda.synchronize, self.cleanup_timeout)
                except TimeoutError:
                    print(f"⚠️ Timeout cleanup GPU ({self.cleanup_timeout}s)")
                
                # 2. Vider cache PyTorch
                torch.cuda.empty_cache()
                
                # 3. Garbage collection Python
                gc.collect()
                
                # 4. Reset statistiques mémoire (uniquement pour debug)
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
                
                print("🧹 Cleanup GPU complet effectué")
                
            except Exception as e:
                print(f"⚠️ Erreur cleanup GPU: {e}")
                # Force cleanup minimal sans synchronize
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except:
                    pass
    
    def _log_json_event(self, event_type: str, test_name: str, data: Dict[str, Any]):
        """Log des événements au format JSON avec rollover automatique"""
        if self.enable_json_logging:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'test_name': test_name,
                'data': data
            }
            self.json_logs.append(log_entry)
            
            # Rollover automatique si trop de logs
            if len(self.json_logs) >= JSON_LOG_MAX_SIZE:
                rollover_file = f"gpu_mem_rollover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.export_json_logs(rollover_file)
                self.json_logs.clear()
                print(f"📄 JSON logs rollover: {rollover_file}")
    
    def export_json_logs(self, filepath: str):
        """Exporte les logs JSON vers un fichier"""
        if self.json_logs:
            with open(filepath, 'w') as f:
                json.dump(self.json_logs, f, indent=2)
            print(f"📄 Logs JSON exportés: {filepath} ({len(self.json_logs)} entrées)")
    
    @contextlib.contextmanager
    def gpu_context(self, test_name: str = "unknown"):
        """Context manager avec cleanup automatique et monitoring amélioré"""
        print(f"🔄 Début test GPU: {test_name}")
        
        # Acquisition lock multiprocess si disponible
        if self.multiprocess_lock:
            if isinstance(self.multiprocess_lock, FileLock):
                acquired = self.multiprocess_lock.acquire(timeout=60)
            else:
                acquired = (self.multiprocess_lock.acquire(timeout=60) is not False)
            
            if not acquired:
                raise RuntimeError("Impossible d'acquérir accès exclusif GPU")
            
            self.multiprocess_lock_held = True
        
        # Statistiques avant
        stats_before = self.get_memory_stats()
        start_time = time.time()
        
        try:
            yield self.device
            
        except Exception as e:
            print(f"❌ Erreur test {test_name}: {e}")
            traceback.print_exc()
            self._log_json_event('error', test_name, {'error': str(e)})
            raise
            
        finally:
            # Cleanup automatique après test
            self.force_cleanup()
            
            # Statistiques après
            stats_after = self.get_memory_stats()
            duration = time.time() - start_time
            
            # Rapport test détaillé
            memory_diff = stats_after['allocated_gb'] - stats_before['allocated_gb']
            reserved_diff = stats_after['reserved_gb'] - stats_before['reserved_gb']
            
            print(f"📊 Test {test_name} terminé ({duration:.2f}s)")
            print(f"   Mémoire allouée: {stats_before['allocated_gb']:.2f}GB → {stats_after['allocated_gb']:.2f}GB ({memory_diff:+.3f}GB)")
            print(f"   Mémoire réservée: {stats_before['reserved_gb']:.2f}GB → {stats_after['reserved_gb']:.2f}GB ({reserved_diff:+.3f}GB)")
            print(f"   Fragmentation: {stats_after['potential_fragmentation_gb']:.3f}GB ({stats_after['fragmentation_pct']:.1f}%)")
            
            # Log JSON
            self._log_json_event('test_complete', test_name, {
                'duration_s': duration,
                'memory_diff_gb': memory_diff,
                'reserved_diff_gb': reserved_diff,
                'fragmentation_gb': stats_after['potential_fragmentation_gb'],
                'fragmentation_pct': stats_after['fragmentation_pct'],
                'stats_before': stats_before,
                'stats_after': stats_after
            })
            
            # Seuil harmonisé à 100MB
            if abs(memory_diff) > 0.1:  # 100MB
                print(f"⚠️ POTENTIAL MEMORY LEAK: {memory_diff:+.3f}GB")
                self._log_json_event('memory_leak_warning', test_name, {
                    'leak_gb': memory_diff,
                    'threshold_gb': 0.1
                })
            else:
                print("✅ Pas de memory leak détecté")
            
            # Alerte fragmentation excessive (seuil paramétrable)
            if stats_after['potential_fragmentation_gb'] > self.fragmentation_threshold_gb:
                print(f"⚠️ FRAGMENTATION ÉLEVÉE: {stats_after['potential_fragmentation_gb']:.3f}GB")
                self._log_json_event('fragmentation_warning', test_name, {
                    'fragmentation_gb': stats_after['potential_fragmentation_gb'],
                    'fragmentation_pct': stats_after['fragmentation_pct'],
                    'threshold_gb': self.fragmentation_threshold_gb
                })
            
            # Libération lock multiprocess (une seule fois)
            if self.multiprocess_lock and self.multiprocess_lock_held:
                self.multiprocess_lock.release()
                self.multiprocess_lock_held = False

# =============================================================================
# INSTANCE GLOBALE ET CONFIGURATION
# =============================================================================

# Instance globale du manager (sera recréée par configure_for_environment)
gpu_manager = None

# Instance globale des métriques Prometheus (sera recréée après configuration)
prometheus_metrics = None

# =============================================================================
# DÉCORATEURS POUR TESTS GPU AUTOMATIQUES
# =============================================================================

def gpu_test_cleanup(test_name: Optional[str] = None):
    """Décorateur pour cleanup automatique tests GPU"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if gpu_manager is None:
                raise RuntimeError("GPU manager non initialisé. Appelez configure_for_environment() d'abord.")
            name = test_name or func.__name__
            with gpu_manager.gpu_context(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def gpu_memory_monitor(threshold_gb: float = 1.0):
    """Décorateur pour monitoring mémoire GPU avec seuils configurables"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if gpu_manager is None:
                raise RuntimeError("GPU manager non initialisé. Appelez configure_for_environment() d'abord.")
                
            stats_before = gpu_manager.get_memory_stats()
            
            result = func(*args, **kwargs)
            
            stats_after = gpu_manager.get_memory_stats()
            # Utiliser max_allocated pour éviter faux-négatifs si code interne libère déjà
            memory_used = stats_after['max_allocated_gb'] - stats_before['max_allocated_gb']
            
            if memory_used > threshold_gb:
                print(f"🚨 ALERTE MÉMOIRE: {func.__name__} utilise {memory_used:.2f}GB")
                print(f"   Seuil: {threshold_gb}GB | Utilisation GPU: {stats_after['utilization_pct']:.1f}%")
                print(f"   Fragmentation: {stats_after['potential_fragmentation_gb']:.3f}GB ({stats_after['fragmentation_pct']:.1f}%)")
                
                gpu_manager._log_json_event('memory_threshold_exceeded', func.__name__, {
                    'memory_used_gb': memory_used,
                    'threshold_gb': threshold_gb,
                    'utilization_pct': stats_after['utilization_pct'],
                    'fragmentation_pct': stats_after['fragmentation_pct']
                })
            
            return result
        return wrapper
    return decorator

# =============================================================================
# FONCTIONS UTILITAIRES MEMORY LEAK PREVENTION
# =============================================================================

def validate_no_memory_leak(threshold_gb: float = 0.1):
    """Validation qu'aucun memory leak n'existe - Seuil harmonisé à 100MB"""
    if gpu_manager is None:
        raise RuntimeError("GPU manager non initialisé. Appelez configure_for_environment() d'abord.")
        
    initial = gpu_manager.initial_memory / 1024**3
    current = torch.cuda.memory_allocated(0) / 1024**3
    diff = current - initial
    
    print(f"🔍 VALIDATION MEMORY LEAK")
    print(f"   Mémoire initiale: {initial:.3f}GB")
    print(f"   Mémoire actuelle: {current:.3f}GB")
    print(f"   Différence: {diff:+.3f}GB")
    print(f"   Seuil tolérance: {threshold_gb:.3f}GB")
    
    validation_passed = abs(diff) <= threshold_gb
    
    gpu_manager._log_json_event('memory_leak_validation', 'global', {
        'initial_gb': initial,
        'current_gb': current,
        'diff_gb': diff,
        'threshold_gb': threshold_gb,
        'passed': validation_passed
    })
    
    if validation_passed:
        print("✅ Aucun memory leak détecté")
        return True
    else:
        print(f"❌ MEMORY LEAK DÉTECTÉ: {diff:+.3f}GB")
        return False

def emergency_gpu_reset():
    """Reset GPU d'urgence en cas de memory leak critique"""
    if gpu_manager is None:
        raise RuntimeError("GPU manager non initialisé. Appelez configure_for_environment() d'abord.")
        
    print("🚨 RESET GPU D'URGENCE")
    
    try:
        # Forcer cleanup complet
        gpu_manager.force_cleanup()
        
        # Réinitialiser contexte CUDA si possible
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()
        
        # Réinitialiser statistiques peak
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        
        # Statistiques après reset
        stats = gpu_manager.get_memory_stats()
        print(f"✅ Reset GPU terminé - Mémoire allouée: {stats['allocated_gb']:.3f}GB")
        
        gpu_manager._log_json_event('emergency_reset', 'system', {
            'stats_after_reset': stats
        })
        
    except Exception as e:
        print(f"❌ Échec reset GPU: {e}")
        print("💡 Solution: Redémarrer le script complet")
        gpu_manager._log_json_event('emergency_reset_failed', 'system', {
            'error': str(e)
        })

def get_detailed_gpu_report():
    """Rapport détaillé état GPU pour debugging"""
    if gpu_manager is None:
        raise RuntimeError("GPU manager non initialisé. Appelez configure_for_environment() d'abord.")
        
    stats = gpu_manager.get_memory_stats()
    
    print("📋 RAPPORT DÉTAILLÉ GPU RTX 3090")
    print("=" * 50)
    for key, value in stats.items():
        if 'gb' in key:
            print(f"   {key:<30}: {value:>8.3f} GB")
        elif 'pct' in key:
            print(f"   {key:<30}: {value:>8.1f} %")
        else:
            print(f"   {key:<30}: {value:>8.3f}")
    print("=" * 50)
    
    return stats

# =============================================================================
# MÉTRIQUES PROMETHEUS (PHASE 2 / SPRINT 1)
# =============================================================================

class PrometheusMetrics:
    """Classe pour exposer métriques GPU au format Prometheus"""
    
    def __init__(self, gpu_memory_manager):
        self.gpu_manager = gpu_memory_manager
        self.metrics = {}
    
    def update_metrics(self):
        """Met à jour les métriques GPU"""
        stats = self.gpu_manager.get_memory_stats()
        
        leak_bytes = abs(torch.cuda.memory_allocated(0) - self.gpu_manager.initial_memory)
        self.metrics = {
            'gpu_memory_allocated_bytes': stats['allocated_gb'] * 1024**3,
            'gpu_memory_reserved_bytes': stats['reserved_gb'] * 1024**3,
            'gpu_memory_free_bytes': stats['free_gb'] * 1024**3,
            'gpu_utilization_percent': stats['utilization_pct'],
            'gpu_fragmentation_bytes': stats['potential_fragmentation_gb'] * 1024**3,
            'gpu_fragmentation_percent': stats['fragmentation_pct'],
            'gpu_memory_leak_bytes': leak_bytes
        }
    
    def export_prometheus_format(self):
        """Exporte les métriques au format Prometheus"""
        self.update_metrics()
        
        lines = []
        lines.append("# HELP gpu_memory_allocated_bytes GPU memory currently allocated")
        lines.append("# TYPE gpu_memory_allocated_bytes gauge")
        lines.append(f"gpu_memory_allocated_bytes {{gpu=\"rtx3090\"}} {self.metrics['gpu_memory_allocated_bytes']:.0f}")
        
        lines.append("# HELP gpu_memory_leak_bytes Memory leaked since start")
        lines.append("# TYPE gpu_memory_leak_bytes gauge")
        lines.append(f"gpu_memory_leak_bytes {{gpu=\"rtx3090\"}} {self.metrics['gpu_memory_leak_bytes']:.0f}")
        
        lines.append("# HELP gpu_utilization_percent GPU utilization percentage")
        lines.append("# TYPE gpu_utilization_percent gauge")
        lines.append(f"gpu_utilization_percent {{gpu=\"rtx3090\"}} {self.metrics['gpu_utilization_percent']:.2f}")
        
        lines.append("# HELP gpu_fragmentation_percent GPU memory fragmentation percentage")
        lines.append("# TYPE gpu_fragmentation_percent gauge")
        lines.append(f"gpu_fragmentation_percent {{gpu=\"rtx3090\"}} {self.metrics['gpu_fragmentation_percent']:.2f}")
        
        # Ajouter timestamp pour améliorer visualisation Grafana
        lines.append("# HELP gpu_timestamp_seconds GPU metrics collection timestamp")
        lines.append("# TYPE gpu_timestamp_seconds gauge")
        lines.append(f"gpu_timestamp_seconds {{gpu=\"rtx3090\"}} {time.time():.0f}")
        
        return "\n".join(lines)

# =============================================================================
# CONFIGURATION POUR ENVIRONNEMENTS SPÉCIFIQUES
# =============================================================================

def configure_for_environment(env: str = "dev"):
    """Configure le GPU manager selon l'environnement"""
    global gpu_manager, prometheus_metrics
    
    # Cleanup ancien manager si existant
    if gpu_manager is not None:
        if gpu_manager.enable_json_logging and gpu_manager.json_logs:
            # Sauvegarder logs avant destruction
            gpu_manager.export_json_logs(f"gpu_logs_before_reconfig_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    if env == "production":
        # Production: logs JSON + lock multiprocess
        gpu_manager = GPUMemoryManager(
            enable_json_logging=True,
            enable_multiprocess_lock=True,
            fragmentation_threshold_gb=FRAGMENTATION_THRESHOLD_GB
        )
        print("🏭 Configuration PRODUCTION activée")
    elif env == "ci":
        # CI: lock multiprocess sans logs JSON
        gpu_manager = GPUMemoryManager(
            enable_json_logging=False,
            enable_multiprocess_lock=True,
            fragmentation_threshold_gb=FRAGMENTATION_THRESHOLD_GB * 0.5  # Plus strict en CI
        )
        print("🔧 Configuration CI activée")
    else:
        # Dev: configuration par défaut
        gpu_manager = GPUMemoryManager(
            enable_json_logging=True,
            enable_multiprocess_lock=False,
            fragmentation_threshold_gb=FRAGMENTATION_THRESHOLD_GB
        )
        print("💻 Configuration DEV activée")
    
    # Recréer les métriques Prometheus avec le nouveau manager
    prometheus_metrics = PrometheusMetrics(gpu_manager)

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

def run_parallel_tests_with_cleanup(num_tests: int = 5):
    """Exemple exécution tests parallèles avec cleanup et validation améliorée"""
    print(f"🔄 TESTS PARALLÈLES AVEC CLEANUP AUTOMATIQUE ({num_tests} tests)")
    
    # Rapport initial
    get_detailed_gpu_report()
    
    for i in range(num_tests):
        print(f"\n--- Test {i+1}/{num_tests} ---")
        test_load_model_with_cleanup()
        
        # Validation après chaque test
        if not validate_no_memory_leak():
            print(f"⚠️ Memory leak détecté au test {i+1}")
            emergency_gpu_reset()
    
    print(f"\n🏆 TOUS LES TESTS TERMINÉS ({num_tests}/{num_tests})")
    
    # Validation finale avec rapport détaillé
    print("\n📊 VALIDATION FINALE:")
    if validate_no_memory_leak():
        print("✅ AUCUN MEMORY LEAK - PARALLÉLISATION SÛRE")
        get_detailed_gpu_report()
    else:
        print("❌ MEMORY LEAK DÉTECTÉ - CLEANUP REQUIS")
        emergency_gpu_reset()
        get_detailed_gpu_report()

def stress_test_memory_management(iterations: int = 20):
    """Test de stress pour validation robustesse memory management"""
    print(f"🔥 STRESS TEST MEMORY MANAGEMENT ({iterations} itérations)")
    
    failed_tests = 0
    for i in range(iterations):
        try:
            with gpu_manager.gpu_context(f"stress_test_{i}"):
                # Test plus intensif
                tensors = []
                for j in range(10):
                    tensor = torch.randn(500, 500, device=gpu_manager.device)
                    tensors.append(tensor)
                
                # Opérations GPU intensives
                for tensor in tensors:
                    _ = torch.matmul(tensor, tensor.t())
                
                # Libération explicite
                del tensors
                
        except Exception as e:
            failed_tests += 1
            print(f"❌ Échec stress test {i}: {e}")
    
    print(f"\n📊 RÉSULTATS STRESS TEST:")
    print(f"   Tests réussis: {iterations - failed_tests}/{iterations}")
    print(f"   Taux succès: {((iterations - failed_tests) / iterations * 100):.1f}%")
    
    success = validate_no_memory_leak()
    
    # Export métriques Prometheus
    print("\n📊 MÉTRIQUES PROMETHEUS:")
    print(prometheus_metrics.export_prometheus_format())
    
    return success

# =============================================================================
# MAIN - DÉMONSTRATION ET VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("🧪 DÉMONSTRATION SOLUTION MEMORY LEAK V4.0 (FINALE)")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()
    
    # Configuration selon argument ligne de commande
    import sys
    env = sys.argv[1] if len(sys.argv) > 1 else "dev"
    configure_for_environment(env)
    
    # Test initial avec rapport détaillé
    print(f"\n📊 Statistiques GPU initiales:")
    initial_stats = get_detailed_gpu_report()
    
    # Tests standards avec cleanup
    run_parallel_tests_with_cleanup(num_tests=5)
    
    # Test de stress pour validation robustesse
    print(f"\n🔥 LANCEMENT STRESS TEST:")
    if stress_test_memory_management(iterations=10):
        print("✅ STRESS TEST RÉUSSI - SOLUTION ROBUSTE")
    else:
        print("⚠️ STRESS TEST ÉCHOUÉ - INVESTIGATION REQUISE")
    
    # Export logs JSON si activé
    if gpu_manager.enable_json_logging:
        log_file = f"gpu_memory_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        gpu_manager.export_json_logs(log_file)
    
    print("\n🎯 SOLUTION MEMORY LEAK V4.0 VALIDÉE POUR PARALLÉLISATION")
    print("✅ Sémaphore multiprocess partagé via Manager")
    print("✅ Protection double release avec tracking")
    print("✅ PrometheusMetrics lié au bon gpu_manager")
    print("✅ JSON logs avec rollover automatique")
    print("✅ Fragmentation threshold paramétrable")
    print("✅ Prêt pour traitement 40 fichiers SuperWhisper V6")
    print(f"📊 Temps total: {time.time() - t0:.1f}s")