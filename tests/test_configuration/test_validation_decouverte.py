#!/usr/bin/env python3
"""
VALIDATION FACTUELLE - DÉCOUVERTE CRITIQUE GPU
Test pour vérifier la configuration GPU réelle du système

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
import subprocess
import sys

def test_gpu_configuration():
    """Test factuel de la configuration GPU"""
    print("🔍 VALIDATION FACTUELLE - CONFIGURATION GPU RÉELLE")
    print("="*60)
    
    # Test 1: Configuration sans CUDA_VISIBLE_DEVICES
    print("\n📊 TEST 1: Configuration GPU native")
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   Nombre de GPUs détectées: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("   ❌ CUDA non disponible")
        return False
    
    # Test 2: Configuration avec CUDA_VISIBLE_DEVICES='1'
    print("\n📊 TEST 2: Configuration avec CUDA_VISIBLE_DEVICES='1'")
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # Force reload CUDA context
    if hasattr(torch.cuda, '_lazy_init'):
        torch.cuda._lazy_init()
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   Nombre de GPUs visibles: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU visible {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("   ❌ CUDA non disponible")
    
    # Test 3: Validation pratique allocation mémoire
    print("\n📊 TEST 3: Test allocation mémoire CUDA:0")
    try:
        # Reset pour test clean
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        device = torch.device('cuda:0')
        test_tensor = torch.randn(1000, 1000).to(device)
        gpu_name = torch.cuda.get_device_name(device)
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        
        print(f"   ✅ Allocation réussie sur: {gpu_name}")
        print(f"   💾 Mémoire allouée: {memory_allocated:.1f}MB")
        
        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ Erreur allocation CUDA:0: {e}")
    
    # Test 4: Validation pratique allocation mémoire
    print("\n📊 TEST 4: Test allocation mémoire CUDA:1")
    try:
        device = torch.device('cuda:1')
        test_tensor = torch.randn(1000, 1000).to(device)
        gpu_name = torch.cuda.get_device_name(device)
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        
        print(f"   ✅ Allocation réussie sur: {gpu_name}")
        print(f"   💾 Mémoire allouée: {memory_allocated:.1f}MB")
        
        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ Erreur allocation CUDA:1: {e}")
    
    return True

def validate_discovery():
    """Valide les affirmations de la découverte critique"""
    print("\n🎯 VALIDATION DES AFFIRMATIONS")
    print("="*40)
    
    # Reset environment
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    if not torch.cuda.is_available():
        print("❌ Impossible de valider - CUDA non disponible")
        return False
    
    try:
        # Vérification 1: RTX 3090 sur CUDA:0
        gpu_0_name = torch.cuda.get_device_name(0)
        gpu_0_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        is_rtx3090_cuda0 = "3090" in gpu_0_name and gpu_0_memory > 20
        print(f"   RTX 3090 sur CUDA:0 ? {'✅ OUI' if is_rtx3090_cuda0 else '❌ NON'}")
        print(f"   GPU 0: {gpu_0_name} ({gpu_0_memory:.1f}GB)")
        
        # Vérification 2: RTX 5060 Ti sur CUDA:1
        if torch.cuda.device_count() > 1:
            gpu_1_name = torch.cuda.get_device_name(1)
            gpu_1_memory = torch.cuda.get_device_properties(1).total_memory / 1024**3
            
            is_rtx5060_cuda1 = "5060" in gpu_1_name and 15 < gpu_1_memory < 17
            print(f"   RTX 5060 Ti sur CUDA:1 ? {'✅ OUI' if is_rtx5060_cuda1 else '❌ NON'}")
            print(f"   GPU 1: {gpu_1_name} ({gpu_1_memory:.1f}GB)")
            
            return is_rtx3090_cuda0 and is_rtx5060_cuda1
        else:
            print("   ❌ Une seule GPU détectée - configuration dual-GPU non confirmée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur validation: {e}")
        return False

if __name__ == "__main__":
    print("🚨 VALIDATION DÉCOUVERTE CRITIQUE GPU")
    print("SuperWhisper V6 - Test Factuel")
    print("="*60)
    
    # Test configuration
    success = test_gpu_configuration()
    
    if success:
        # Validation des affirmations
        discovery_valid = validate_discovery()
        
        print(f"\n🎯 RÉSULTAT FINAL:")
        print(f"   Découverte critique fondée ? {'✅ OUI' if discovery_valid else '❌ NON'}")
        
        if discovery_valid:
            print("   ✅ Les configurations suggérées sont correctes:")
            print("   ✅ CUDA:0 = RTX 3090 (24GB) = SÉCURISÉ")
            print("   ✅ CUDA:1 = RTX 5060 Ti (16GB) = À ÉVITER")
        else:
            print("   ❌ La découverte critique nécessite révision")
    else:
        print("\n❌ Impossible de valider la configuration GPU") 