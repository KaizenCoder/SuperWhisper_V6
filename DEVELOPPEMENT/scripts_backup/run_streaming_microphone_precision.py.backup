#!/usr/bin/env python3
"""
Streaming microphone HAUTE PRÉCISION - SuperWhisper V6
Utilise le modèle LARGE-V2 pour une transcription fidèle
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

print("🎮 Streaming Microphone PRÉCISION - Configuration GPU RTX 3090 (CUDA:1)")
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
    """Démonstration haute précision avec modèle large-v2"""
    print("🚀 Streaming Microphone HAUTE PRÉCISION - SuperWhisper V6")
    print("=" * 60)
    
    # Validation GPU RTX 3090
    try:
        validate_rtx3090_configuration()
    except Exception as e:
        print(f"❌ Erreur configuration GPU: {e}")
        return
    
    print("\n🔧 Initialisation HAUTE PRÉCISION (modèle large-v2)...")
    
    try:
        # Configuration pour modèle LARGE-V2 (précision maximale)
        precision_config = {
            'fallback_chain': ['prism_large_v2'],  # SEUL le modèle large-v2
            'cache_size_mb': 200,                  # Cache plus important
            'cache_ttl': 600,                      # TTL cache 10 minutes
            'timeout_per_minute': 5.0,             # Timeout plus généreux
            'retry_attempts': 3,                   # Plus de tentatives
            'enable_fallback': False               # Pas de fallback, précision pure
        }
        
        # Initialisation STT Manager en mode précision
        print("⚡ Initialisation UnifiedSTTManager (large-v2)...")
        print("⏳ Chargement du modèle large-v2... (peut prendre 30-60s)")
        stt_mgr = UnifiedSTTManager(precision_config)
        print("✅ UnifiedSTTManager initialisé")
        
        # Initialisation Streaming Microphone Manager
        print("⚡ Initialisation StreamingMicrophoneManager...")
        mic_mgr = StreamingMicrophoneManager(stt_mgr)
        print("✅ StreamingMicrophoneManager initialisé")
        
        print("\n🎤 Démarrage capture microphone HAUTE PRÉCISION...")
        print("🗣️ Parlez maintenant! (Ctrl+C pour arrêter)")
        print("🎯 Objectifs: Transcription fidèle à 95%+")
        print("📊 Modèle: large-v2 (précision maximale)")
        print("⚡ Latence: ~1-2s (normal pour la précision)")
        print("-" * 60)
        
        # Démarrage streaming
        await mic_mgr.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
        print("🎉 Test streaming microphone HAUTE PRÉCISION terminé")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        # Diagnostic si erreur
        print("\n🔍 DIAGNOSTIC:")
        print("1. Le modèle large-v2 nécessite plus de VRAM")
        print("2. Essayez: python scripts/run_streaming_microphone_fast.py")
        print("3. Vérifiez la mémoire GPU disponible")
    
    print("\n🎯 Streaming microphone haute précision terminé")

if __name__ == "__main__":
    print("🎯 Démarrage streaming microphone HAUTE PRÉCISION...")
    
    try:
        asyncio.run(main())
        print("\n✅ Session streaming haute précision terminée avec succès")
    except KeyboardInterrupt:
        print("\n🛑 Session interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        sys.exit(1) 