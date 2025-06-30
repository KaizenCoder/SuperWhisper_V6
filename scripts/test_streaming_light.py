#!/usr/bin/env python3
"""
Test streaming microphone léger - SuperWhisper V6
Utilise le modèle small pour un test rapide

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
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 Test Streaming Léger - Configuration GPU RTX 3090 (CUDA:1)")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports
import torch
import sounddevice as sd
import webrtcvad
import numpy as np

class LightStreamingTest:
    """Test streaming léger sans modèles lourds"""
    
    def __init__(self):
        self.frame_count = 0
        self.speech_detected = 0
        
    def test_audio_capture(self, duration=5):
        """Test simple de capture audio"""
        print(f"🎤 Test capture audio ({duration}s)...")
        
        sample_rate = 16000
        frames_collected = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"⚠️ Status: {status}")
            frames_collected.append(indata.copy())
        
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                callback=callback
            ):
                print("🔴 Enregistrement... parlez maintenant!")
                time.sleep(duration)
                
            total_frames = len(frames_collected)
            print(f"✅ Capture terminée: {total_frames} frames collectées")
            return True
            
        except Exception as e:
            print(f"❌ Erreur capture: {e}")
            return False
    
    def test_vad_processing(self):
        """Test traitement VAD simple"""
        print("🔍 Test VAD...")
        
        try:
            vad = webrtcvad.Vad(2)
            
            # Test avec silence
            silence = np.zeros(320, dtype=np.int16)  # 20ms à 16kHz
            is_speech_silence = vad.is_speech(silence.tobytes(), 16000)
            
            # Test avec bruit simulé
            noise = (np.random.random(320) * 1000).astype(np.int16)
            is_speech_noise = vad.is_speech(noise.tobytes(), 16000)
            
            print(f"✅ VAD silence: {is_speech_silence} (doit être False)")
            print(f"✅ VAD bruit: {is_speech_noise}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur VAD: {e}")
            return False
    
    def test_streaming_simulation(self, duration=10):
        """Simulation streaming sans STT"""
        print(f"🔄 Simulation streaming ({duration}s)...")
        
        sample_rate = 16000
        frame_duration_ms = 20
        frame_size = sample_rate * frame_duration_ms // 1000
        
        try:
            vad = webrtcvad.Vad(2)
            speech_frames = []
            silence_count = 0
            max_silence = 20  # 400ms de silence
            
            def process_frame(indata, frames, time, status):
                self.frame_count += 1
                
                # Convertir en int16
                frame_int16 = (indata[:, 0] * 32767).astype(np.int16)
                
                # VAD
                is_speech = vad.is_speech(frame_int16.tobytes(), sample_rate)
                
                if is_speech:
                    speech_frames.append(frame_int16)
                    silence_count = 0
                    self.speech_detected += 1
                    print(f"🗣️ Parole détectée (frame {self.frame_count})")
                else:
                    silence_count += 1
                    
                    # Fin d'énoncé détectée
                    if speech_frames and silence_count >= max_silence:
                        total_audio = np.concatenate(speech_frames)
                        duration_ms = len(total_audio) / sample_rate * 1000
                        print(f"📝 Énoncé terminé: {duration_ms:.0f}ms, {len(speech_frames)} frames")
                        
                        # Reset pour prochain énoncé
                        speech_frames.clear()
                        silence_count = 0
            
            with sd.InputStream(
                samplerate=sample_rate,
                blocksize=frame_size,
                channels=1,
                dtype='float32',
                callback=process_frame
            ):
                print("🔴 Streaming actif... parlez!")
                time.sleep(duration)
            
            print(f"✅ Streaming terminé: {self.frame_count} frames, {self.speech_detected} avec parole")
            return True
            
        except Exception as e:
            print(f"❌ Erreur streaming: {e}")
            return False

def test_device_list():
    """Liste les périphériques audio"""
    print("🎤 Périphériques audio:")
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} ({device['max_input_channels']} ch)")
        return True
    except Exception as e:
        print(f"❌ Erreur liste périphériques: {e}")
        return False

def test_gpu_basic():
    """Test GPU basique"""
    print("🔍 Test GPU...")
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("❌ CUDA non disponible")
            return False
    except Exception as e:
        print(f"❌ Erreur GPU: {e}")
        return False

async def main():
    """Test principal léger"""
    print("🚀 Test Streaming Microphone Léger - SuperWhisper V6")
    print("=" * 50)
    
    # Tests séquentiels
    tests = [
        ("GPU", test_gpu_basic),
        ("Périphériques", test_device_list),
        ("VAD", lambda: LightStreamingTest().test_vad_processing()),
        ("Capture Audio", lambda: LightStreamingTest().test_audio_capture(3)),
        ("Streaming Simulation", lambda: LightStreamingTest().test_streaming_simulation(10))
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Test {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: RÉUSSI")
            else:
                print(f"❌ {test_name}: ÉCHOUÉ")
                
        except Exception as e:
            print(f"💥 {test_name}: ERREUR - {e}")
            results.append((test_name, False))
    
    # Résultats finaux
    print("\n📊 Résultats finaux:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 TOUS LES TESTS LÉGERS RÉUSSIS!")
        print("✅ Prêt pour test avec STT complet")
        return True
    else:
        print("⚠️ Certains tests ont échoué")
        print("🔧 Correction nécessaire avant STT")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        
        if result:
            print("\n🎉 VALIDATION LÉGÈRE RÉUSSIE!")
            print("🚀 Procédez au test complet avec: python scripts/run_streaming_microphone_demo.py")
        else:
            print("\n❌ VALIDATION LÉGÈRE ÉCHOUÉE")
            print("🔧 Corrigez les problèmes avant de continuer")
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrompu")
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        import traceback
        traceback.print_exc() 