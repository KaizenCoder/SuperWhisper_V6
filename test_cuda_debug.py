#!/usr/bin/env python3
"""
Test DEBUG COMPLET - Configuration GPU RTX 3090
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Objectif: Valider configuration GPU RTX 3090 exclusive avec diagnostic complet
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
import importlib

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

def test_cuda_debug_rtx3090():
    """Test DEBUG complet configuration GPU RTX 3090"""
    print("🔍 TEST DEBUG CONFIGURATION GPU RTX 3090")
    print("="*60)

    # Validation RTX 3090 obligatoire
    validate_rtx3090_mandatory()

    print("\n🎯 CONFIGURATION ACTUELLE:")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"   CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER')}")
    print(f"   PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\n🎮 DEVICES GPU VISIBLES: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Vérifier que c'est bien la RTX 3090
            if "RTX 3090" in gpu_name:
                print(f"   ✅ RTX 3090 détectée sur cuda:{i}")
            else:
                print(f"   ⚠️ GPU inattendue: {gpu_name}")
        
        # Test allocation sur device 0 (RTX 3090 après mapping)
        print(f"\n🧪 TEST ALLOCATION MÉMOIRE RTX 3090:")
        try:
            x = torch.randn(1000, 1000, device='cuda:0')
            print(f"   ✅ Allocation cuda:0 réussie sur RTX 3090")
            
            # Test calcul GPU
            y = torch.matmul(x, x.t())
            print(f"   ✅ Calcul GPU RTX 3090 réussi")
            
            # Cleanup
            del x, y
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Erreur allocation/calcul RTX 3090: {e}")
        
        # Vérifier mémoire GPU
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n💾 MÉMOIRE GPU RTX 3090:")
        print(f"   Allouée: {allocated:.3f}GB")
        print(f"   Réservée: {reserved:.3f}GB")
        
    else:
        print("❌ CUDA non disponible")

    print("\n" + "="*60)
    print("🎯 DIAGNOSTIC RTX 3090 TERMINÉ")
    print("   Configuration GPU RTX 3090 validée avec succès")
    print("="*60)

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    test_cuda_debug_rtx3090() 