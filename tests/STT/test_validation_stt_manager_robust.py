#!/usr/bin/env python3
"""
VALIDATION FACTUELLE - STT/stt_manager_robust.py
Test pour vérifier que le manager utilise RTX 3090 (CUDA:0)

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
import asyncio
import logging
import os

# Test de la configuration RTX 3090
def test_stt_manager_gpu_config():
    """Test factuel de la configuration GPU du STT manager"""
    print("🔍 VALIDATION - STT/stt_manager_robust.py")
    print("="*50)
    
    # Nettoyer variables environnement pour test propre
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    # Configuration test
    config = {
        'use_gpu': True,
        'use_vad': False,
        'model_cache_dir': './models'
    }
    
    try:
        # Import et instanciation 
        sys.path.append('.')
        from STT.stt_manager_robust import RobustSTTManager
        
        manager = RobustSTTManager(config)
        print(f"✅ STTManager instancié avec device: {manager.device}")
        
        # Vérifier device GPU
        if manager.device == "cuda":
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            vram_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            print(f"   Device actuel: {current_device}")
            print(f"   GPU utilisée: {gpu_name}")
            print(f"   VRAM totale: {vram_total:.1f}GB")
            
            is_rtx3090 = "3090" in gpu_name
            print(f"   RTX 3090 confirmée: {'✅ OUI' if is_rtx3090 else '❌ NON'}")
            
            return is_rtx3090 and current_device == 0
            
        else:
            print(f"   ❌ Device non-GPU: {manager.device}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur instanciation STT manager: {e}")
        return False

def test_gpu_allocation_direct():
    """Test direct d'allocation GPU"""
    print("\n🎮 TEST ALLOCATION GPU DIRECTE")
    print("="*40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible")
        return False
    
    try:
        # Test CUDA:0 direct
        device = torch.device('cuda:0')
        test_tensor = torch.randn(100, 100).to(device)
        gpu_name = torch.cuda.get_device_name(device)
        
        print(f"   Device testé: cuda:0")
        print(f"   GPU utilisée: {gpu_name}")
        
        is_rtx3090 = "3090" in gpu_name
        print(f"   RTX 3090 confirmée: {'✅ OUI' if is_rtx3090 else '❌ NON'}")
        
        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()
        
        return is_rtx3090
        
    except Exception as e:
        print(f"❌ Erreur allocation: {e}")
        return False

async def test_manager_initialization():
    """Test d'initialisation asynchrone du manager"""
    print("\n🚀 TEST INITIALISATION MANAGER")
    print("="*40)
    
    # Nettoyer variables environnement
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    try:
        sys.path.append('.')
        from STT.stt_manager_robust import RobustSTTManager
        
        config = {
            'use_gpu': True,
            'use_vad': False,
            'model_cache_dir': './models'
        }
        
        manager = RobustSTTManager(config)
        print(f"   ✅ Manager créé sur device: {manager.device}")
        
        if manager.device == "cuda":
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"   GPU active: {gpu_name} (device {current_device})")
            
            is_correct = "3090" in gpu_name and current_device == 0
            print(f"   Configuration correcte: {'✅ OUI' if is_correct else '❌ NON'}")
            return is_correct
        else:
            print(f"   ❌ Device non-GPU: {manager.device}")
            return False
        
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        return False

if __name__ == "__main__":
    print("🚨 VALIDATION STT MANAGER - RTX 3090")
    print("="*60)
    
    # Tests
    config_valid = test_stt_manager_gpu_config()
    gpu_valid = test_gpu_allocation_direct()
    init_valid = asyncio.run(test_manager_initialization())
    
    # Résultat final
    print(f"\n🎯 RÉSULTAT FINAL:")
    print(f"   Configuration GPU: {'✅' if config_valid else '❌'}")
    print(f"   Allocation directe: {'✅' if gpu_valid else '❌'}")
    print(f"   Initialisation: {'✅' if init_valid else '❌'}")
    
    overall_success = config_valid and gpu_valid and init_valid
    print(f"   Validation globale: {'✅ RÉUSSIE' if overall_success else '❌ ÉCHEC'}")
    
    if overall_success:
        print("   ✅ stt_manager_robust.py utilise correctement RTX 3090")
    else:
        print("   ❌ stt_manager_robust.py nécessite correction") 