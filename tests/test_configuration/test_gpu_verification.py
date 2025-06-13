#!/usr/bin/env python3
"""
Test de vérification GPU RTX 3090
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
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

def test_gpu_verification_rtx3090():
    """Test de vérification complète GPU RTX 3090"""
    print("=== TEST VÉRIFICATION GPU RTX 3090 ===")
    
    # Validation RTX 3090 obligatoire
    validate_rtx3090_mandatory()
    
    print(f"🎯 CUDA disponible: {torch.cuda.is_available()}")
    print(f"🔢 Nombre de GPU visibles: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        # Vérification de tous les GPU visibles (devrait être seulement RTX 3090)
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            gpu_memory = props.total_memory / 1024**3
            
            print(f"\n🎮 GPU {i}:")
            print(f"   Nom: {gpu_name}")
            print(f"   Mémoire: {gpu_memory:.1f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Multiprocesseurs: {props.multi_processor_count}")
            
            # Vérification RTX 3090 exclusive
            if "RTX 3090" in gpu_name:
                print(f"   ✅ RTX 3090 confirmée sur device {i}")
            else:
                print(f"   ❌ GPU inattendue: {gpu_name}")
                raise RuntimeError(f"GPU incorrecte détectée: {gpu_name}")
        
        print("\n=== TEST AVEC CONFIGURATION ACTUELLE ===")
        print(f"🔒 CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
        print(f"🔧 CUDA_DEVICE_ORDER = '{os.environ.get('CUDA_DEVICE_ORDER')}'")
        print(f"⚡ PYTORCH_CUDA_ALLOC_CONF = '{os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}'")
        
        # Test device principal (RTX 3090 mappée en cuda:0)
        main_gpu = torch.cuda.get_device_name(0)
        main_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n🏆 GPU principale mappée cuda:0: {main_gpu}")
        print(f"💾 Mémoire disponible: {main_memory:.1f} GB")
        
        # Test allocation mémoire
        print(f"\n🧪 TEST ALLOCATION MÉMOIRE RTX 3090:")
        try:
            x = torch.randn(2000, 2000, device='cuda:0')  # 16MB test
            print(f"   ✅ Allocation 16MB réussie sur RTX 3090")
            
            # Test calcul GPU
            y = torch.matmul(x, x.t())
            print(f"   ✅ Calcul matriciel RTX 3090 réussi")
            
            # Test device correct
            print(f"   📍 Device utilisé: {x.device}")
            
            # Cleanup
            del x, y
            torch.cuda.empty_cache()
            print(f"   ✅ Nettoyage mémoire effectué")
            
        except Exception as e:
            print(f"   ❌ Erreur allocation RTX 3090: {e}")
            raise
        
        # Statistiques mémoire finales
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n💾 MÉMOIRE RTX 3090 - Allouée: {allocated:.3f}GB, Réservée: {reserved:.3f}GB")
        
    else:
        print("❌ CUDA non disponible")
        raise RuntimeError("CUDA non disponible")
    
    print("\n" + "="*50)
    print("✅ VÉRIFICATION GPU RTX 3090 TERMINÉE AVEC SUCCÈS")
    print("   Configuration RTX 3090 exclusive validée")
    print("="*50)

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    test_gpu_verification_rtx3090() 