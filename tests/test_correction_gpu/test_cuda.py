#!/usr/bin/env python3
"""
Test de détection CUDA avec PyTorch
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

import torch

def validate_rtx3090_mandatory():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER', '')
    if cuda_order != 'PCI_BUS_ID':
        raise RuntimeError(f"🚫 CUDA_DEVICE_ORDER='{cuda_order}' incorrect - doit être 'PCI_BUS_ID'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

def test_cuda_rtx3090():
    """Test CUDA RTX 3090 avec validation complète"""
    print("🚨 RTX 5060 Ti MASQUÉE / RTX 3090 devient device 0 visible")
    print("=== TEST RTX 3090 EXCLUSIF ===")
    print(f"🎯 CUDA disponible: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"🔥 Nombre de GPU visibles: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"\n   GPU {i}: {gpu_name}")
            print(f"   Mémoire: {gpu_memory:.1f} GB")
            
            # Validation RTX 3090 exclusive
            if "RTX 3090" in gpu_name and gpu_memory >= 20:
                print(f"   ✅ RTX 3090 confirmée sur device {i}")
            elif "RTX 5060" in gpu_name:
                print(f"   🚫 RTX 5060 Ti détectée - DEVRAIT ÊTRE MASQUÉE!")
                raise RuntimeError("RTX 5060 Ti non masquée - configuration GPU incorrecte")
            
            # Test d'allocation sur RTX 3090
            if "RTX 3090" in gpu_name:
                try:
                    torch.cuda.set_device(i)
                    x = torch.randn(3000, 3000, device=f'cuda:{i}')  # Test 36MB sur RTX 3090
                    print(f"   ✅ Allocation 36MB RTX 3090 réussie!")
                    print(f"   📊 Tensor sur: {x.device}")
                    
                    # Test calcul GPU
                    y = torch.matmul(x, x.t())
                    print(f"   ✅ Calcul matriciel RTX 3090 réussi")
                    
                    # Cleanup mémoire
                    del x, y
                    torch.cuda.empty_cache()
                    print(f"   ✅ Nettoyage mémoire RTX 3090 effectué")
                    
                except Exception as e:
                    print(f"   ❌ Erreur allocation RTX 3090: {e}")
                    raise
        
        print(f"\n🎯 Version CUDA: {torch.version.cuda}")
        print(f"🎯 GPU courant: {torch.cuda.current_device()}")
        
        # Statistiques mémoire finales
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"💾 Mémoire RTX 3090 - Allouée: {allocated:.3f}GB, Réservée: {reserved:.3f}GB")

    else:
        print("❌ CUDA non disponible")
        raise RuntimeError("CUDA non disponible")

    print("\n" + "="*50)
    print("✅ Test CUDA RTX 3090 terminé avec succès")

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    test_cuda_rtx3090() 