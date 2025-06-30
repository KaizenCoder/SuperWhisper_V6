#!/usr/bin/env python3
"""
Streaming microphone MEDIUM - SuperWhisper V6
Utilise le modèle MEDIUM pour un bon compromis vitesse/précision
"""

import os
import sys
import asyncio
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 Streaming Microphone MEDIUM - Configuration GPU RTX 3090 (CUDA:1)")
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
    """Démonstration avec modèle medium (compromis optimal)"""
    print("🚀 Streaming Microphone MEDIUM - SuperWhisper V6")
    print("=" * 55)
    
    # Validation GPU RTX 3090
    try:
        validate_rtx3090_configuration()
    except Exception as e:
        print(f"❌ Erreur configuration GPU: {e}")
        return
    
    print("\n🔧 Initialisation MEDIUM (compromis vitesse/précision)...")
    
    try:
        # Configuration pour modèle MEDIUM (compromis optimal)
        medium_config = {
            'fallback_chain': ['prism_medium'],  # SEUL le modèle medium
            'cache_size_mb': 100,                # Cache équilibré
            'cache_ttl': 450,                    # TTL cache 7.5 minutes
            'timeout_per_minute': 3.0,           # Timeout équilibré
            'retry_attempts': 2,                 # Tentatives raisonnables
            'enable_fallback': True              # Fallback activé
        }
        
        # Initialisation STT Manager en mode medium
        print("⚡ Initialisation UnifiedSTTManager (medium)...")
        print("⏳ Chargement du modèle medium... (10-20s)")
        stt_mgr = UnifiedSTTManager(medium_config)
        print("✅ UnifiedSTTManager initialisé")
        
        # Initialisation Streaming Microphone Manager
        print("⚡ Initialisation StreamingMicrophoneManager...")
        mic_mgr = StreamingMicrophoneManager(stt_mgr)
        print("✅ StreamingMicrophoneManager initialisé")
        
        print("\n🎤 Démarrage capture microphone MEDIUM...")
        print("🗣️ Parlez maintenant! (Ctrl+C pour arrêter)")
        print("🎯 Objectifs: Bon compromis précision/vitesse")
        print("📊 Modèle: medium (équilibré)")
        print("⚡ Latence: ~800ms-1.2s (optimal)")
        print("🎯 Précision: ~85-90% (très bonne)")
        print("-" * 55)
        
        # Démarrage streaming
        await mic_mgr.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
        print("🎉 Test streaming microphone MEDIUM terminé")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        # Diagnostic si erreur
        print("\n🔍 DIAGNOSTIC:")
        print("1. Le modèle medium est un bon compromis")
        print("2. Si problème, essayez: python scripts/run_streaming_microphone_fast.py")
        print("3. Vérifiez la mémoire GPU disponible")
    
    print("\n🎯 Streaming microphone medium terminé")

if __name__ == "__main__":
    print("🎯 Démarrage streaming microphone MEDIUM...")
    
    try:
        asyncio.run(main())
        print("\n✅ Session streaming medium terminée avec succès")
    except KeyboardInterrupt:
        print("\n🛑 Session interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        sys.exit(1) 