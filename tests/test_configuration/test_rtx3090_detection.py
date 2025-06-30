#!/usr/bin/env python3
"""
Test détection GPU RTX 3090 - Configuration double GPU
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

def test_gpu_detection():
    """Test détection GPU avec configuration RTX 3090"""
    print("🔍 TEST DÉTECTION GPU RTX 3090")
    print("=" * 40)
    
    # Test PyTorch avec validation RTX 3090
    try:
        validate_rtx3090_mandatory()
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)  # Device 0 visible = RTX 3090
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            compute_cap = torch.cuda.get_device_capability(0)
            
            print(f"🎮 GPU détecté: {gpu_name}")
            print(f"💾 VRAM: {gpu_memory:.1f}GB")
            print(f"🔧 Compute Capability: {compute_cap}")
            
            # Vérifier si c'est RTX 3090
            is_rtx_3090 = "RTX 3090" in gpu_name and gpu_memory >= 20
            print(f"🏆 RTX 3090 détecté: {'✅ OUI' if is_rtx_3090 else '❌ NON'}")
            
            if not is_rtx_3090:
                raise RuntimeError(f"GPU incorrecte détectée: {gpu_name}")
            
            return is_rtx_3090
        else:
            print("❌ CUDA non disponible")
            raise RuntimeError("CUDA non disponible")
            
    except ImportError as e:
        print(f"❌ PyTorch non installé: {e}")
        raise
    except Exception as e:
        print(f"❌ Erreur PyTorch: {e}")
        raise

def test_faster_whisper():
    """Test faster-whisper avec RTX 3090"""
    print("\n🎤 TEST FASTER-WHISPER RTX 3090")
    print("=" * 40)
    
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper importé")
        
        # Test initialisation GPU RTX 3090 - faster-whisper utilise "cuda" générique
        print("🔄 Test initialisation GPU RTX 3090...")
        model = WhisperModel("tiny", device="cuda", compute_type="int8")
        print("✅ Modèle Whisper GPU RTX 3090 initialisé")
        
        # Test transcription rapide
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)
        segments, info = model.transcribe(dummy_audio)
        segments_list = list(segments)  # Force exécution
        
        print("✅ Test transcription réussi")
        print(f"📊 Langue détectée: {info.language}")
        print(f"🔢 Segments traités: {len(segments_list)}")
        
        # Cleanup mémoire
        del model
        torch.cuda.empty_cache()
        print("✅ Nettoyage mémoire RTX 3090 effectué")
        
        return True
        
    except ImportError as e:
        print(f"❌ faster-whisper non installé: {e}")
        print("   Installer avec: pip install faster-whisper")
        return False
    except Exception as e:
        print(f"❌ Erreur faster-whisper: {e}")
        print(f"   Détail: {type(e).__name__}: {str(e)}")
        return False

def main():
    """Fonction principale de diagnostic RTX 3090"""
    print("🚀 DIAGNOSTIC COMPLET RTX 3090")
    print("Configuration: RTX 3090 (CUDA:1) exclusive - CORRECTION CRITIQUE")
    print("⚠️  ANCIENNE CONFIG RTX 5060 Ti SUPPRIMÉE - RTX 3090 exclusive maintenant")
    print()
    
    try:
        # Test 1: Détection GPU
        gpu_ok = test_gpu_detection()
        
        # Test 2: faster-whisper si GPU OK
        whisper_ok = False
        if gpu_ok:
            whisper_ok = test_faster_whisper()
        
        # Résumé
        print("\n📋 RÉSUMÉ DIAGNOSTIC RTX 3090")
        print("=" * 40)
        print(f"🎮 GPU RTX 3090: {'✅ OK' if gpu_ok else '❌ ÉCHEC'}")
        print(f"🎤 faster-whisper: {'✅ OK' if whisper_ok else '❌ ÉCHEC'}")
        
        if gpu_ok and whisper_ok:
            print("\n🎉 CONFIGURATION RTX 3090 OPÉRATIONNELLE")
            print("   SuperWhisper V6 peut maintenant fonctionner")
        else:
            print("\n🚨 PROBLÈME CONFIGURATION RTX 3090")
            if not gpu_ok:
                print("   - Vérifier installation CUDA/PyTorch")
            if gpu_ok and not whisper_ok:
                print("   - Problème faster-whisper ou CUDA libraries")
        
        return gpu_ok and whisper_ok
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        return False

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    main() 