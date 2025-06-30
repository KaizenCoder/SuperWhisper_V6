#!/usr/bin/env python3
"""
Streaming microphone RAPIDE - SuperWhisper V6
Utilise le modèle SMALL pour un démarrage rapide

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

import asyncio
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 Streaming Microphone RAPIDE - Configuration GPU RTX 3090 (CUDA:1)")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports après configuration GPU
import torch
from STT.unified_stt_manager import UnifiedSTTManager
from STT.streaming_microphone_manager import StreamingMicrophoneManager

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

async def main():
    """Démonstration rapide avec modèle small"""
    print("🚀 Streaming Microphone RAPIDE - SuperWhisper V6")
    print("=" * 50)
    
    # Validation GPU RTX 3090
    try:
        validate_rtx3090_configuration()
    except Exception as e:
        print(f"❌ Erreur configuration GPU: {e}")
        return
    
    print("\n🔧 Initialisation RAPIDE (modèle small)...")
    
    try:
        # Configuration pour modèle SMALL uniquement
        fast_config = {
            'fallback_chain': ['prism_small'],  # SEUL le modèle small
            'cache_size_mb': 50,                # Cache réduit en MB
            'cache_ttl': 300,                   # TTL cache 5 minutes
            'timeout_per_minute': 2.0,          # Timeout réduit
            'retry_attempts': 2,                # Moins de tentatives
            'enable_fallback': True             # Fallback activé
        }
        
        # Initialisation STT Manager en mode rapide
        print("⚡ Initialisation UnifiedSTTManager (small)...")
        stt_mgr = UnifiedSTTManager(fast_config)
        print("✅ UnifiedSTTManager initialisé")
        
        # Initialisation Streaming Microphone Manager
        print("⚡ Initialisation StreamingMicrophoneManager...")
        mic_mgr = StreamingMicrophoneManager(stt_mgr)
        print("✅ StreamingMicrophoneManager initialisé")
        
        print("\n🎤 Démarrage capture microphone RAPIDE...")
        print("🗣️ Parlez maintenant! (Ctrl+C pour arrêter)")
        print("🎯 Objectifs: Premier token < 800ms, Latence moyenne < 1s")
        print("📊 Modèle: small (démarrage rapide)")
        print("-" * 50)
        
        # Démarrage streaming
        await mic_mgr.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
        print("🎉 Test streaming microphone TERMINÉ")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        # Diagnostic si erreur
        print("\n🔍 DIAGNOSTIC:")
        print("1. Vérifiez que le microphone fonctionne")
        print("2. Essayez: python scripts/test_streaming_light.py")
        print("3. Vérifiez la configuration GPU RTX 3090")
    
    print("\n🎯 Streaming microphone rapide terminé")

if __name__ == "__main__":
    print("🎯 Démarrage streaming microphone RAPIDE...")
    
    try:
        asyncio.run(main())
        print("\n✅ Session streaming terminée avec succès")
    except KeyboardInterrupt:
        print("\n🛑 Session interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        sys.exit(1) 