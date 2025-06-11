#!/usr/bin/env python3
"""
Test d'accès RTX 3090
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

def test_rtx3090_access():
    """Test d'accès et fonctionnalité RTX 3090"""
    print("=== TEST ACCÈS RTX 3090 ===")
    
    # Validation RTX 3090 obligatoire
    validate_rtx3090_mandatory()
    
    print(f"🔒 CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
    print(f"🔧 CUDA_DEVICE_ORDER = '{os.environ.get('CUDA_DEVICE_ORDER')}'")
    print(f"🎯 CUDA disponible: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"🔢 Nombre de GPU visibles: {device_count}")
        
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_memory = props.total_memory / 1024**3
        
        print(f"🎮 GPU 0 (RTX 3090 mappée): {gpu_name}")
        print(f"💾 Mémoire: {gpu_memory:.1f} GB")
        print(f"🔧 Compute Capability: {props.major}.{props.minor}")
        
        # Vérification RTX 3090
        if "RTX 3090" not in gpu_name:
            raise RuntimeError(f"GPU incorrecte détectée: {gpu_name}")
        
        # Test création tensor simple
        print("\n🧪 Test création tensor sur RTX 3090...")
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
            print(f"   ✅ Tensor créé sur: {x.device}")
            print(f"   📊 Valeurs: {x.tolist()}")
            
            # Test calcul GPU
            y = x * 2
            print(f"   ✅ Calcul GPU réussi: {y.tolist()}")
            
            # Test allocation plus importante
            z = torch.randn(1000, 1000, device='cuda:0')
            print(f"   ✅ Allocation 4MB RTX 3090 réussie")
            
            # Test opération matricielle
            w = torch.matmul(z, z.t())
            print(f"   ✅ Multiplication matricielle RTX 3090 réussie")
            print(f"   📏 Taille résultat: {w.shape}")
            
            # Cleanup
            del x, y, z, w
            torch.cuda.empty_cache()
            print(f"   ✅ Nettoyage mémoire RTX 3090 effectué")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            raise
        
        # Statistiques mémoire
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n💾 MÉMOIRE RTX 3090 - Allouée: {allocated:.3f}GB, Réservée: {reserved:.3f}GB")
        
    else:
        print("❌ CUDA non disponible!")
        raise RuntimeError("CUDA non disponible")
    
    print("\n" + "="*40)
    print("✅ TEST ACCÈS RTX 3090 RÉUSSI")
    print("   Accès et fonctionnalité RTX 3090 validés")
    print("="*40)

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    test_rtx3090_access() 