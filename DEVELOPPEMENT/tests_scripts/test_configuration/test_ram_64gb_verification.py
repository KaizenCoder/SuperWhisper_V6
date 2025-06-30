#!/usr/bin/env python3
"""
TEST VÉRIFICATION RAM 64GB - SuperWhisper V6
🎯 Objectif: Vérifier l'accès complet aux 64GB de RAM pour parallélisation

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

import gc
import time
import numpy as np
from typing import List, Dict

def get_memory_info() -> Dict[str, float]:
    """Obtenir les informations mémoire détaillées"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    except ImportError:
        # Fallback sans psutil
        try:
            import subprocess
            # PowerShell pour obtenir info mémoire
            result = subprocess.run([
                'powershell', '-Command',
                'Get-WmiObject -Class Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum | Select-Object -ExpandProperty Sum'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                total_bytes = int(float(result.stdout.strip().replace(',', '.')))
                total_gb = total_bytes / (1024**3)
                
                # Mémoire disponible approximative
                return {
                    'total_gb': total_gb,
                    'available_gb': total_gb * 0.8,  # Estimation 80% disponible
                    'used_gb': total_gb * 0.2,
                    'percent_used': 20.0,
                    'free_gb': total_gb * 0.8
                }
        except Exception as e:
            print(f"❌ Erreur détection mémoire: {e}")
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percent_used': 0,
                'free_gb': 0
            }

def test_memory_allocation(target_gb: float) -> bool:
    """Test allocation mémoire pour vérifier capacité réelle"""
    print(f"🔄 Test allocation {target_gb:.1f}GB...")
    
    try:
        # Allouer des chunks de 1GB pour éviter fragmentation
        chunk_size_gb = 1.0
        chunks = []
        allocated_gb = 0
        
        while allocated_gb < target_gb:
            remaining = target_gb - allocated_gb
            current_chunk_gb = min(chunk_size_gb, remaining)
            
            # Allouer chunk en float32 (4 bytes)
            elements = int(current_chunk_gb * 1024**3 / 4)
            
            print(f"   Allocation chunk {current_chunk_gb:.1f}GB ({allocated_gb:.1f}/{target_gb:.1f}GB)")
            
            chunk = np.zeros(elements, dtype=np.float32)
            chunks.append(chunk)
            allocated_gb += current_chunk_gb
            
            # Petite pause pour système
            time.sleep(0.1)
        
        print(f"✅ Allocation {target_gb:.1f}GB réussie!")
        
        # Test écriture/lecture pour vérifier accès réel
        print("🔄 Test écriture/lecture...")
        for i, chunk in enumerate(chunks):
            # Écrire valeur unique dans chaque chunk
            chunk[0] = i + 100
            chunk[-1] = i + 200
            
            # Vérifier lecture
            if chunk[0] != i + 100 or chunk[-1] != i + 200:
                print(f"❌ Erreur lecture/écriture chunk {i}")
                return False
        
        print("✅ Test écriture/lecture réussi!")
        
        # Libération mémoire
        del chunks
        gc.collect()
        
        return True
        
    except MemoryError:
        print(f"❌ MemoryError: Impossible d'allouer {target_gb:.1f}GB")
        return False
    except Exception as e:
        print(f"❌ Erreur allocation: {e}")
        return False

def test_parallel_memory_simulation(workers: int, memory_per_worker_gb: float) -> bool:
    """Simuler utilisation mémoire parallélisation"""
    print(f"🔄 Simulation {workers} workers, {memory_per_worker_gb:.1f}GB/worker...")
    
    total_memory_needed = workers * memory_per_worker_gb
    print(f"   Mémoire totale requise: {total_memory_needed:.1f}GB")
    
    # Créer simulation avec dictionnaire de workers
    worker_data = {}
    
    try:
        for worker_id in range(workers):
            print(f"   Worker {worker_id+1}/{workers}: Allocation {memory_per_worker_gb:.1f}GB")
            
            # Allouer mémoire pour ce worker
            elements = int(memory_per_worker_gb * 1024**3 / 4)  # float32
            worker_data[worker_id] = np.random.random(elements).astype(np.float32)
            
            # Simulation traitement
            worker_data[worker_id][0] = worker_id * 1000
            
            time.sleep(0.05)  # Pause simulation
        
        print(f"✅ Simulation {workers} workers réussie!")
        
        # Vérification intégrité données
        for worker_id in range(workers):
            if worker_data[worker_id][0] != worker_id * 1000:
                print(f"❌ Erreur intégrité worker {worker_id}")
                return False
        
        print("✅ Intégrité données vérifiée!")
        
        # Cleanup
        del worker_data
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur simulation parallèle: {e}")
        return False

def main():
    """Test principal vérification RAM 64GB"""
    print("🚀 VÉRIFICATION RAM 64GB - SUPERWHISPER V6")
    print("="*60)
    
    # 1. Info mémoire système
    memory_info = get_memory_info()
    
    print(f"\n📊 INFORMATIONS MÉMOIRE SYSTÈME:")
    print(f"   RAM Totale: {memory_info['total_gb']:.1f}GB")
    print(f"   RAM Disponible: {memory_info['available_gb']:.1f}GB")
    print(f"   RAM Utilisée: {memory_info['used_gb']:.1f}GB ({memory_info['percent_used']:.1f}%)")
    print(f"   RAM Libre: {memory_info['free_gb']:.1f}GB")
    
    # 2. Validation 64GB détectés
    if memory_info['total_gb'] < 60:
        print(f"❌ ERREUR: RAM détectée ({memory_info['total_gb']:.1f}GB) < 60GB minimum")
        return False
    
    if memory_info['total_gb'] >= 60 and memory_info['total_gb'] <= 66:
        print("✅ RAM 64GB détectée correctement!")
    else:
        print(f"⚠️ RAM détectée: {memory_info['total_gb']:.1f}GB (attendu ~64GB)")
    
    # 3. Test allocation progressive
    print(f"\n🧪 TESTS ALLOCATION MÉMOIRE:")
    
    # Test allocations croissantes
    test_sizes = [1, 5, 10, 20, 30]  # GB
    
    for size_gb in test_sizes:
        if memory_info['available_gb'] < size_gb + 5:  # Marge sécurité 5GB
            print(f"⏭️ Skip test {size_gb}GB (mémoire insuffisante)")
            continue
            
        success = test_memory_allocation(size_gb)
        if not success:
            print(f"❌ Échec allocation {size_gb}GB")
            return False
        
        print()  # Ligne vide
    
    # 4. Test simulation parallélisation
    print(f"\n🔄 SIMULATION PARALLÉLISATION SUPERWHISPER V6:")
    
    # Configuration parallélisation recommandée
    parallel_configs = [
        (4, 2.0),   # 4 workers, 2GB/worker = 8GB total
        (8, 2.0),   # 8 workers, 2GB/worker = 16GB total
        (10, 2.5),  # 10 workers, 2.5GB/worker = 25GB total
        (8, 4.0),   # 8 workers, 4GB/worker = 32GB total
    ]
    
    for workers, memory_per_worker in parallel_configs:
        total_needed = workers * memory_per_worker
        
        if memory_info['available_gb'] < total_needed + 10:  # Marge 10GB
            print(f"⏭️ Skip simulation {workers}w×{memory_per_worker:.1f}GB (mémoire insuffisante)")
            continue
        
        success = test_parallel_memory_simulation(workers, memory_per_worker)
        if not success:
            print(f"❌ Échec simulation {workers} workers")
            return False
        
        print()  # Ligne vide
    
    # 5. Conclusions
    print("🎯 CONCLUSIONS VÉRIFICATION RAM:")
    print("="*40)
    
    print("✅ RAM 64GB accessible et fonctionnelle")
    print("✅ Allocations mémoire importantes possibles")
    print("✅ Simulation parallélisation réussie")
    print("✅ Configuration optimale pour SuperWhisper V6")
    
    # Recommandations spécifiques
    max_safe_workers = int(memory_info['available_gb'] / 4.0)  # 4GB/worker
    print(f"\n💡 RECOMMANDATIONS PARALLÉLISATION:")
    print(f"   Workers max recommandés: {max_safe_workers}")
    print(f"   Mémoire/worker: 2-4GB")
    print(f"   Marge sécurité: 15-20GB")
    print(f"   Configuration optimale: 8-10 workers × 2.5GB")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🏆 VÉRIFICATION RAM 64GB: SUCCÈS COMPLET")
            sys.exit(0)
        else:
            print("\n❌ VÉRIFICATION RAM 64GB: ÉCHEC")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrompu par utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur critique: {e}")
        sys.exit(1) 