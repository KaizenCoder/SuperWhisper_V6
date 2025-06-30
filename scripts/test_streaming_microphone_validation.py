#!/usr/bin/env python3
"""
Test de validation streaming microphone SuperWhisper V6
Solution des experts - Validation complète RTX 3090

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
import logging
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 Test Streaming Microphone - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports après configuration GPU
import torch
import numpy as np
import sounddevice as sd
from STT.unified_stt_manager import UnifiedSTTManager
from STT.streaming_microphone_manager import StreamingMicrophoneManager, TranscriptionResult

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def list_audio_devices():
    """Liste les périphériques audio disponibles"""
    print("\n🎤 Périphériques audio disponibles:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (max_input_channels: {device['max_input_channels']})")
    print()

class TestStreamingMicrophone:
    """Classe de test pour le streaming microphone"""
    
    def __init__(self):
        self.stt_manager = None
        self.mic_manager = None
        self.transcriptions = []
        self.start_time = None
        
    async def setup(self):
        """Initialisation des managers"""
        print("🔧 Initialisation UnifiedSTTManager...")
        self.stt_manager = UnifiedSTTManager()
        
        print("🔧 Initialisation StreamingMicrophoneManager...")
        self.mic_manager = StreamingMicrophoneManager(self.stt_manager)
        
        # Override du callback pour collecter les résultats
        self.mic_manager._on_transcription = self._collect_transcription
        
        print("✅ Managers initialisés")
    
    def _collect_transcription(self, result: TranscriptionResult):
        """Collecte les transcriptions pour analyse"""
        if self.start_time is None:
            self.start_time = time.time()
        
        self.transcriptions.append({
            'text': result.text,
            'latency_ms': result.latency_ms,
            'timestamp': time.time() - self.start_time,
            'start_ms': result.start_ms,
            'end_ms': result.end_ms
        })
        
        # Affichage en temps réel
        print(f"⏱️ {result.latency_ms:.1f}ms | {result.text}")
        
        # Validation latence (objectif < 800ms)
        if result.latency_ms > 800:
            print(f"⚠️ Latence élevée: {result.latency_ms:.1f}ms (objectif: <800ms)")
        else:
            print(f"✅ Latence OK: {result.latency_ms:.1f}ms")
    
    async def test_basic_streaming(self, duration_seconds: int = 30):
        """Test de base du streaming microphone"""
        print(f"\n🎤 Test streaming microphone ({duration_seconds}s)")
        print("Parlez maintenant... (Ctrl+C pour arrêter)")
        
        try:
            # Démarrer le streaming avec timeout
            await asyncio.wait_for(
                self.mic_manager.run(),
                timeout=duration_seconds
            )
        except asyncio.TimeoutError:
            print(f"⏰ Test terminé après {duration_seconds}s")
        except KeyboardInterrupt:
            print("🛑 Test interrompu par l'utilisateur")
    
    def analyze_results(self):
        """Analyse des résultats de test"""
        if not self.transcriptions:
            print("❌ Aucune transcription collectée")
            return False
        
        print(f"\n📊 Analyse des résultats ({len(self.transcriptions)} transcriptions)")
        
        # Statistiques latence
        latencies = [t['latency_ms'] for t in self.transcriptions]
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"⏱️ Latence moyenne: {avg_latency:.1f}ms")
        print(f"⏱️ Latence max: {max_latency:.1f}ms")
        print(f"⏱️ Latence min: {min_latency:.1f}ms")
        
        # Validation objectifs
        success_criteria = []
        
        # 1. Premier mot < 800ms
        if self.transcriptions:
            first_latency = self.transcriptions[0]['latency_ms']
            if first_latency <= 800:
                print(f"✅ Premier mot: {first_latency:.1f}ms (≤800ms)")
                success_criteria.append(True)
            else:
                print(f"❌ Premier mot: {first_latency:.1f}ms (>800ms)")
                success_criteria.append(False)
        
        # 2. Transcriptions non vides
        non_empty = [t for t in self.transcriptions if t['text'].strip()]
        if len(non_empty) > 0:
            print(f"✅ Transcriptions non vides: {len(non_empty)}/{len(self.transcriptions)}")
            success_criteria.append(True)
        else:
            print(f"❌ Aucune transcription non vide")
            success_criteria.append(False)
        
        # 3. Latence moyenne acceptable
        if avg_latency <= 1000:  # Objectif élargi pour validation
            print(f"✅ Latence moyenne acceptable: {avg_latency:.1f}ms")
            success_criteria.append(True)
        else:
            print(f"❌ Latence moyenne trop élevée: {avg_latency:.1f}ms")
            success_criteria.append(False)
        
        # Résultat global
        success_rate = sum(success_criteria) / len(success_criteria) * 100
        print(f"\n🎯 Taux de réussite: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 TEST RÉUSSI - Streaming microphone fonctionnel!")
            return True
        else:
            print("❌ TEST ÉCHOUÉ - Problèmes détectés")
            return False

async def test_device_detection():
    """Test de détection des périphériques"""
    print("🔍 Test détection périphériques audio...")
    
    try:
        # Test sounddevice
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if input_devices:
            print(f"✅ {len(input_devices)} périphériques d'entrée détectés")
            return True
        else:
            print("❌ Aucun périphérique d'entrée détecté")
            return False
            
    except Exception as e:
        print(f"❌ Erreur détection périphériques: {e}")
        return False

async def test_vad_realtime():
    """Test VAD en temps réel (simulation)"""
    print("🔍 Test VAD temps réel...")
    
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)  # Aggressiveness 2
        
        # Test avec silence (simulation)
        silence = np.zeros(320, dtype=np.int16)  # 20ms à 16kHz
        is_speech = vad.is_speech(silence.tobytes(), 16000)
        
        if not is_speech:
            print("✅ VAD détecte correctement le silence")
            return True
        else:
            print("❌ VAD détecte incorrectement de la parole dans le silence")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test VAD: {e}")
        return False

async def main():
    """Fonction principale de test"""
    print("🚀 Test de validation streaming microphone SuperWhisper V6")
    print("=" * 60)
    
    # Validation GPU RTX 3090
    try:
        validate_rtx3090_configuration()
    except Exception as e:
        print(f"❌ Erreur configuration GPU: {e}")
        return False
    
    # Liste des périphériques
    list_audio_devices()
    
    # Tests préliminaires
    print("\n📋 Tests préliminaires...")
    
    device_ok = await test_device_detection()
    vad_ok = await test_vad_realtime()
    
    if not (device_ok and vad_ok):
        print("❌ Tests préliminaires échoués")
        return False
    
    print("✅ Tests préliminaires réussis")
    
    # Test principal streaming
    print("\n🎤 Test principal streaming microphone...")
    
    tester = TestStreamingMicrophone()
    
    try:
        await tester.setup()
        
        # Test de 30 secondes
        await tester.test_basic_streaming(30)
        
        # Analyse des résultats
        success = tester.analyze_results()
        
        if success:
            print("\n🎉 VALIDATION STREAMING MICROPHONE RÉUSSIE!")
            print("✅ Phase 4 STT peut être marquée comme COMPLÉTÉE")
            return True
        else:
            print("\n❌ VALIDATION STREAMING MICROPHONE ÉCHOUÉE")
            print("🔧 Ajustements nécessaires avant finalisation")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 Démarrage test streaming microphone...")
    
    try:
        result = asyncio.run(main())
        
        if result:
            print("\n🎉 MISSION ACCOMPLIE - STREAMING MICROPHONE VALIDÉ!")
            sys.exit(0)
        else:
            print("\n❌ MISSION ÉCHOUÉE - PROBLÈMES DÉTECTÉS")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 