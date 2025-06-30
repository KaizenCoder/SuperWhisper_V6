#!/usr/bin/env python3
"""
Fix TTS Phase 3 - Restaurer Configuration Validée
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Utilise la configuration exacte de la Phase 3 qui avait des performances record :
- Latence Cache : 29.5ms (objectif <100ms)
- Taux Cache : 93.1% (objectif >80%)
- Throughput : 174.9 chars/s (objectif >100)
- Stabilité : 100% (objectif >95%)

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
import time

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Imports
sys.path.insert(0, '.')
from TTS.tts_manager import UnifiedTTSManager
import sounddevice as sd
import numpy as np
import wave
import io

async def test_tts_phase3_config():
    """
    Test TTS avec configuration exacte Phase 3 validée
    """
    print("\n🎯 RÉPARATION TTS - CONFIGURATION PHASE 3 VALIDÉE")
    print("=" * 60)
    
    # Configuration exacte Phase 3 (performances record)
    config = {
        'enable_piper_native': False,  # Désactivé car problématique
        'cache': {
            'max_size_mb': 100,
            'ttl_seconds': 3600
        },
        'circuit_breaker': {
            'failure_threshold': 3,
            'reset_timeout_seconds': 60
        },
        'backends': {
            'piper_cli': {
                'enabled': True,
                'executable_path': 'piper/piper.exe',
                'model_path': 'models/fr_FR-siwis-medium.onnx',
                'speaker_id': 0,
                'target_latency_ms': 1000,
                'sample_rate': 22050,
                'channels': 1,
                'use_json_config': True,
                'length_scale': 1.0
            },
            'sapi_french': {
                'enabled': True,
                'voice_name': 'Microsoft Hortense Desktop',
                'rate': 0,
                'volume': 100,
                'target_latency_ms': 500
            },
            'silent_emergency': {
                'enabled': True,
                'target_latency_ms': 10
            }
        },
        'advanced': {
            'max_text_length': 1000,
            'sample_rate': 22050,
            'channels': 1
        }
    }
    
    try:
        print("🔧 Initialisation TTS avec config Phase 3...")
        tts = UnifiedTTSManager(config)
        print("✅ TTS initialisé avec succès")
        
        # Test 1: Synthèse sans cache (forcer backend réel)
        test_text = f"Test SuperWhisper V6 - {time.time()}"  # Texte unique pour éviter cache
        print(f"\\n🗣️ Test synthèse (sans cache): '{test_text}'")
        
        start_time = time.perf_counter()
        result = await tts.synthesize(test_text, reuse_cache=False)  # FORCER SANS CACHE
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        print(f"\\n📊 RÉSULTATS TTS PHASE 3:")
        print(f"✅ Succès: {result.success}")
        print(f"🔧 Backend utilisé: {result.backend_used}")
        print(f"⏱️ Latence backend: {result.latency_ms:.1f}ms")
        print(f"⏱️ Latence totale: {latency_ms:.1f}ms")
        
        if result.success and result.backend_used != 'cache':
            print(f"🎊 SUCCÈS ! Backend réel utilisé: {result.backend_used}")
            
            if result.audio_data:
                print(f"🔊 Audio généré: {len(result.audio_data)} bytes")
                
                # Test lecture audio
                try:
                    # Conversion bytes → numpy pour lecture
                    audio_io = io.BytesIO(result.audio_data)
                    with wave.open(audio_io, 'rb') as wav_file:
                        frames = wav_file.readframes(-1)
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        sampwidth = wav_file.getsampwidth()
                        
                    print(f"📊 Format audio: {sample_rate}Hz, {channels}ch, {sampwidth*8}bit")
                    
                    # Conversion pour sounddevice
                    if sampwidth == 2:  # 16-bit
                        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        audio_np = np.frombuffer(frames, dtype=np.float32)
                    
                    if channels == 1:
                        audio_np = audio_np.reshape(-1, 1)
                    
                    print(f"🎵 Lecture audio ({len(audio_np)} samples)...")
                    sd.play(audio_np, samplerate=sample_rate)
                    sd.wait()  # Attendre fin lecture
                    print("✅ Audio joué avec succès")
                    
                    # Validation humaine
                    response = input("\\n🗣️ Avez-vous entendu une VRAIE VOIX (pas un bip) ? (o/n): ").lower().strip()
                    
                    if response == 'o':
                        print("🎊 PARFAIT ! TTS Phase 3 restauré avec succès")
                        print("🔧 Configuration validée pour pipeline complet")
                        return True, config
                    else:
                        print("❌ Audio non satisfaisant - diagnostic requis")
                        return False, config
                        
                except Exception as e:
                    print(f"❌ Erreur lecture audio: {e}")
                    return False, config
            else:
                print("❌ Aucun audio généré")
                return False, config
        else:
            print(f"❌ ÉCHEC ! Backend: {result.backend_used}, Erreur: {result.error}")
            return False, config
            
    except Exception as e:
        print(f"❌ Erreur TTS: {e}")
        return False, config
    finally:
        if 'tts' in locals():
            await tts.cleanup()

async def main():
    print("🚀 RÉPARATION TTS - RESTAURATION CONFIGURATION PHASE 3")
    print("Performances record attendues: 29.5ms latence, 93.1% cache hit")
    
    success, config = await test_tts_phase3_config()
    
    if success:
        print("\\n🎊 TTS PHASE 3 RESTAURÉ AVEC SUCCÈS")
        print("✅ Configuration validée pour pipeline complet")
        print("🔧 Prêt pour validation humaine complète")
        
        # Sauvegarder configuration validée
        import yaml
        config_path = "PIPELINE/config/tts_phase3_validated.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"💾 Configuration sauvegardée: {config_path}")
        
        return True
    else:
        print("\\n❌ TTS PHASE 3 ÉCHEC - DIAGNOSTIC APPROFONDI REQUIS")
        print("🔍 Vérifier backends disponibles et configuration")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 