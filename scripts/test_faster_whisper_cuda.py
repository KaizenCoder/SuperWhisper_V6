#!/usr/bin/env python3
"""
Test Diagnostic - faster-whisper + CUDA RTX 3090
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch

def test_torch_cuda():
    """Test configuration PyTorch CUDA"""
    print("\n🔍 TEST PYTORCH CUDA")
    print("-" * 40)
    
    print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ Nombre de GPU: {torch.cuda.device_count()}")
        print(f"✅ GPU actuel: {torch.cuda.current_device()}")
        print(f"✅ Nom GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

def test_faster_whisper_import():
    """Test import faster-whisper"""
    print("\n📦 TEST IMPORT FASTER-WHISPER")
    print("-" * 40)
    
    try:
        from faster_whisper import WhisperModel
        print("✅ Import faster_whisper réussi")
        return True
    except ImportError as e:
        print(f"❌ Import faster_whisper échoué: {e}")
        return False

def test_faster_whisper_devices():
    """Test devices disponibles pour faster-whisper"""
    print("\n🎮 TEST DEVICES FASTER-WHISPER")
    print("-" * 40)
    
    try:
        from faster_whisper import WhisperModel
        
        # Test différentes configurations device
        devices_to_test = [
            "cuda",
            "cuda:0", 
            "cuda:1",
            "cpu"
        ]
        
        for device in devices_to_test:
            try:
                print(f"🧪 Test device '{device}'...")
                model = WhisperModel("tiny", device=device, compute_type="float16")
                print(f"   ✅ Device '{device}' fonctionne")
                del model  # Libérer mémoire
                return device
            except Exception as e:
                print(f"   ❌ Device '{device}' échoue: {e}")
        
        print("❌ Aucun device CUDA fonctionnel")
        return None
        
    except Exception as e:
        print(f"❌ Erreur test devices: {e}")
        return None

def test_faster_whisper_model():
    """Test création modèle faster-whisper"""
    print("\n🤖 TEST MODÈLE FASTER-WHISPER")
    print("-" * 40)
    
    try:
        from faster_whisper import WhisperModel
        
        # Test avec device auto-détecté
        print("🚀 Création modèle tiny (test rapide)...")
        
        # Essayer d'abord avec device auto
        try:
            model = WhisperModel("tiny", device="auto", compute_type="float16")
            print("✅ Modèle créé avec device='auto'")
            device_info = model.device
            print(f"   Device utilisé: {device_info}")
            del model
            return True
        except Exception as e:
            print(f"❌ Device 'auto' échoue: {e}")
        
        # Essayer avec CPU en fallback
        try:
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("✅ Modèle créé avec device='cpu'")
            del model
            return True
        except Exception as e:
            print(f"❌ Device 'cpu' échoue aussi: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur création modèle: {e}")
        return False

def main():
    """Test principal diagnostic"""
    print("🎯 DIAGNOSTIC FASTER-WHISPER + CUDA RTX 3090")
    print("=" * 60)
    
    # 1. Test PyTorch CUDA
    test_torch_cuda()
    
    # 2. Test import faster-whisper
    import_ok = test_faster_whisper_import()
    if not import_ok:
        print("\n❌ ARRÊT: faster-whisper non importable")
        return False
    
    # 3. Test devices
    working_device = test_faster_whisper_devices()
    
    # 4. Test modèle
    model_ok = test_faster_whisper_model()
    
    # 5. Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DIAGNOSTIC")
    print("="*60)
    
    print(f"✅ PyTorch CUDA: {torch.cuda.is_available()}")
    print(f"✅ faster-whisper import: {import_ok}")
    print(f"{'✅' if working_device else '❌'} Device fonctionnel: {working_device or 'Aucun'}")
    print(f"{'✅' if model_ok else '❌'} Modèle créable: {model_ok}")
    
    if working_device and model_ok:
        print("\n🎉 DIAGNOSTIC POSITIF!")
        print(f"   → Utiliser device='{working_device}' dans backend optimisé")
    else:
        print("\n⚠️ PROBLÈMES DÉTECTÉS")
        print("   → Vérifier installation CUDA/faster-whisper")
    
    return working_device and model_ok

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ Diagnostic terminé avec succès")
    else:
        print(f"\n❌ Diagnostic révèle des problèmes") 