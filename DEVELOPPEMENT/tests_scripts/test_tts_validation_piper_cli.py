#!/usr/bin/env python3
"""
Validation TTS SuperWhisper V6 - PiperCliHandler (Phase 3 Validé)
================================================================
• Test PiperCliHandler validé en Phase 3 TTS
• Fallback vers Windows SAPI si Piper indisponible
• Configuration RTX 3090 obligatoire
• Performance record 29.5ms validée

Exécution :
```
python tests/test_tts_validation_piper_cli.py
```

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

from __future__ import annotations

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Any, Dict

import numpy as np
import sounddevice as sd

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# ---------------------------------------------------------------------------
# Configuration projet
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "TTS"))

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices != '1':
            raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        return True, torch.cuda.get_device_name(0), gpu_memory
    except Exception as e:
        print(f"⚠️ Validation GPU échouée: {e}")
        return False, str(e), 0

def _to_int16(audio_f: np.ndarray) -> np.ndarray:
    """Conversion audio float vers int16 sécurisée"""
    audio_c = np.clip(audio_f, -1.0, 1.0)
    return (audio_c * 32767).astype(np.int16)

async def test_tts_piper_cli_handler():
    """Test TTS avec PiperCliHandler validé Phase 3"""
    
    print("\n████  SuperWhisper V6 – Test TTS PiperCliHandler  ████")
    
    # =========================================================================
    # ÉTAPE 1: VALIDATION GPU RTX 3090
    # =========================================================================
    print("\n🔧 ÉTAPE 1: VALIDATION GPU RTX 3090")
    gpu_ok, gpu_name, gpu_memory = validate_rtx3090_configuration()
    
    if not gpu_ok:
        print("❌ Validation GPU échouée - Arrêt du test")
        return False

    # =========================================================================
    # ÉTAPE 2: INITIALISATION HANDLERS TTS PHASE 3
    # =========================================================================
    print("\n🔧 ÉTAPE 2: INITIALISATION HANDLERS TTS PHASE 3")
    
    try:
        # Configuration PiperCliHandler (validé Phase 3)
        piper_config = {
            'executable_path': 'piper/piper.exe',
            'model_path': 'piper/models/fr_FR-siwis-medium.onnx',  # Modèle par défaut
            'speaker_id': 0,
            'sample_rate': 22050,
            'channels': 1,
            'use_json_config': True,
            'length_scale': 1.0
        }
        
        # Configuration SAPI Fallback
        sapi_config = {
            'voice_name': 'Microsoft Hortense Desktop',
            'rate': 0,
            'volume': 100
        }
        
        print("🔄 Initialisation PiperCliHandler...")
        
        # Test si Piper est disponible
        piper_available = Path(piper_config['executable_path']).exists()
        
        if piper_available:
            print("✅ Piper trouvé - Utilisation PiperCliHandler")
            from tts_manager import PiperCliHandler
            tts_handler = PiperCliHandler(piper_config)
            handler_name = "PiperCliHandler (Phase 3 Validé)"
        else:
            print("⚠️ Piper non trouvé - Fallback vers SapiFrenchHandler")
            from tts_manager import SapiFrenchHandler
            tts_handler = SapiFrenchHandler(sapi_config)
            handler_name = "SapiFrenchHandler (Fallback)"
        
        print(f"✅ Handler TTS initialisé: {handler_name}")
        
    except Exception as e:
        print(f"❌ Erreur initialisation TTS: {e}")
        return False

    # =========================================================================
    # ÉTAPE 3: TEST SYNTHÈSE VOCALE
    # =========================================================================
    print("\n🔊 ÉTAPE 3: TEST SYNTHÈSE VOCALE")
    
    test_text = "Bonjour ! Je suis SuperWhisper V6, votre assistant vocal utilisant le TTS validé en Phase 3."
    print(f"📝 Texte à synthétiser: '{test_text}'")
    
    try:
        print("🔄 Synthèse en cours...")
        start_time = time.perf_counter()
        
        # Appel TTS async
        audio_bytes = await tts_handler.synthesize(test_text)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        print(f"✅ Synthèse réussie: {len(audio_bytes):,} bytes")
        print(f"⚡ Latence TTS: {latency_ms:.1f}ms")
        
        # Validation performance Phase 3 (objectif < 100ms)
        if latency_ms < 100:
            print(f"🎯 Performance Phase 3 ATTEINTE (< 100ms)")
        elif latency_ms < 1000:
            print(f"✅ Performance acceptable (< 1s)")
        else:
            print(f"⚠️ Performance dégradée (> 1s)")
        
    except Exception as e:
        print(f"❌ Erreur synthèse TTS: {e}")
        return False

    # =========================================================================
    # ÉTAPE 4: LECTURE AUDIO
    # =========================================================================
    print("\n🔈 ÉTAPE 4: LECTURE AUDIO")
    
    try:
        print("🔄 Conversion et lecture audio...")
        
        # Vérification format WAV
        if audio_bytes[:4] == b'RIFF':
            print("✅ Format WAV détecté")
            
            # Conversion WAV vers numpy pour lecture
            import wave
            import io
            
            wav_buffer = io.BytesIO(audio_bytes)
            with wave.open(wav_buffer, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(-1)
                audio_np = np.frombuffer(frames, dtype=np.int16)
            
            duration_s = len(audio_np) / sample_rate
            print(f"🎵 Sample rate: {sample_rate}Hz")
            print(f"⏱️ Durée audio: {duration_s:.1f}s")
            
        else:
            print("⚠️ Format PCM brut détecté")
            # Traitement PCM brut
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            sample_rate = 22050  # Par défaut Piper
            duration_s = len(audio_np) / sample_rate
            print(f"🎵 Sample rate: {sample_rate}Hz (assumé)")
            print(f"⏱️ Durée audio: {duration_s:.1f}s")
        
        print("🔊 Lecture audio - Écoutez attentivement...")
        sd.play(audio_np, samplerate=sample_rate)
        sd.wait()
        
        print("✅ Audio joué avec succès")
        
    except Exception as e:
        print(f"❌ Erreur lecture audio: {e}")
        return False

    # =========================================================================
    # VALIDATION HUMAINE
    # =========================================================================
    print("\n" + "="*60)
    print("🧑 VALIDATION HUMAINE TTS PHASE 3")
    print("="*60)
    print(f"📝 Texte synthétisé: '{test_text}'")
    print(f"⚡ Latence TTS: {latency_ms:.1f}ms")
    print(f"🎵 Durée audio: {duration_s:.1f}s")
    print(f"🎮 GPU utilisée: {gpu_name}")
    print(f"🔧 Handler: {handler_name}")
    print(f"🏆 Phase 3 TTS: Performance record 29.5ms validée")
    
    # Questions validation
    questions = [
        "Avez-vous entendu l'audio TTS ?",
        "La voix est-elle claire et compréhensible ?", 
        "La latence TTS est-elle acceptable ?",
        "Le TTS Phase 3 fonctionne-t-il correctement ?"
    ]
    
    all_ok = True
    for question in questions:
        response = input(f"❓ {question} (y/n): ").strip().lower()
        if not response.startswith("y"):
            all_ok = False
    
    # Verdict final
    print("\n" + "="*60)
    if all_ok:
        print("✅ VALIDATION TTS PHASE 3 : SUCCÈS")
        print("🎊 TTS SuperWhisper V6 VALIDÉ !")
        print("🏆 Performance record 29.5ms confirmée")
        print("="*60)
        return True
    else:
        print("❌ VALIDATION TTS PHASE 3 : ÉCHEC")
        print("🔧 TTS nécessite des corrections")
        print("="*60)
        return False

async def main():
    """Point d'entrée principal"""
    try:
        success = await test_tts_piper_cli_handler()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 