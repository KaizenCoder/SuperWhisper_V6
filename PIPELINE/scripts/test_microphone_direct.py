#!/usr/bin/env python3
"""
Test Microphone Direct SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Test direct microphone → STT → LLM → TTS → Audio

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
from datetime import datetime
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter les répertoires au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))  # PIPELINE
sys.path.append(str(Path(__file__).parent.parent.parent))  # Racine

try:
    from STT.streaming_microphone_manager import StreamingMicrophoneManager
    from STT.unified_stt_manager import UnifiedSTTManager
    from LLM.llm_client import LLMClient
    from TTS.unified_tts_manager import UnifiedTTSManager
    from PIPELINE.audio_output_manager import AudioOutputManager
    import torch
    import httpx
    import yaml
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    print("💡 Vérifiez que tous les modules sont disponibles")
    sys.exit(1)

def validate_rtx3090():
    """Validation obligatoire RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

async def test_microphone_streaming():
    """Test streaming microphone temps réel"""
    print("\n🎤 TEST MICROPHONE STREAMING TEMPS RÉEL")
    print("=" * 50)
    
    try:
        # Initialiser le microphone manager
        print("🔧 Initialisation StreamingMicrophoneManager...")
        mic_manager = StreamingMicrophoneManager()
        
        # Variables pour capturer les transcriptions
        transcriptions = []
        
        def on_transcription(text, confidence=None):
            """Callback pour les transcriptions"""
            timestamp = datetime.now().strftime("%H:%M:%S")
            transcriptions.append({
                "timestamp": timestamp,
                "text": text,
                "confidence": confidence
            })
            print(f"📝 [{timestamp}] Transcrit: '{text}'")
            if confidence:
                print(f"    Confiance: {confidence:.2f}")
        
        # Configurer le callback
        mic_manager.set_transcription_callback(on_transcription)
        
        print("\n🎤 Microphone prêt !")
        print("🗣️ Parlez maintenant pendant 15 secondes...")
        print("⏹️ Le système va transcrire en temps réel")
        print("💡 Appuyez sur Ctrl+C pour arrêter plus tôt")
        
        # Démarrer l'écoute
        await mic_manager.start_streaming()
        
        # Écouter pendant 15 secondes
        try:
            await asyncio.sleep(15)
        except KeyboardInterrupt:
            print("\n⚠️ Arrêt demandé par l'utilisateur")
        
        # Arrêter l'écoute
        await mic_manager.stop_streaming()
        
        # Résultats
        print(f"\n📊 RÉSULTATS MICROPHONE")
        print(f"🎤 Transcriptions capturées: {len(transcriptions)}")
        
        if transcriptions:
            print("\n📝 Transcriptions détaillées:")
            for i, trans in enumerate(transcriptions, 1):
                print(f"  {i}. [{trans['timestamp']}] {trans['text']}")
            
            # Texte complet
            texte_complet = " ".join([t['text'] for t in transcriptions])
            print(f"\n💬 Texte complet: '{texte_complet}'")
            
            return texte_complet
        else:
            print("⚠️ Aucune transcription capturée")
            return None
            
    except Exception as e:
        print(f"❌ Erreur microphone: {e}")
        return None

async def test_llm_response(texte_input):
    """Test génération réponse LLM"""
    if not texte_input:
        print("⚠️ Pas de texte pour le LLM")
        return None
    
    print(f"\n🤖 TEST LLM AVEC INPUT: '{texte_input}'")
    
    try:
        # Initialiser client LLM
        llm_client = LLMClient(
            endpoint="http://localhost:11434/api/chat",
            model="nous-hermes-2-mistral-7b-dpo:latest",
            timeout=30.0
        )
        
        # Générer réponse
        debut = time.time()
        reponse = await llm_client.generate_response(texte_input)
        fin = time.time()
        
        latence = (fin - debut) * 1000
        
        if reponse:
            print(f"🤖 Réponse LLM: '{reponse}'")
            print(f"⏱️ Latence LLM: {latence:.1f}ms")
            return reponse
        else:
            print("❌ Pas de réponse du LLM")
            return None
            
    except Exception as e:
        print(f"❌ Erreur LLM: {e}")
        return None

async def test_tts_synthesis(texte_reponse):
    """Test synthèse TTS"""
    if not texte_reponse:
        print("⚠️ Pas de texte pour le TTS")
        return None
    
    print(f"\n🔊 TEST TTS AVEC TEXTE: '{texte_reponse}'")
    
    try:
        # Initialiser TTS manager
        tts_manager = UnifiedTTSManager()
        
        # Synthétiser
        debut = time.time()
        result = await tts_manager.synthesize_speech(texte_reponse)
        fin = time.time()
        
        latence = (fin - debut) * 1000
        
        if result and result.success:
            print(f"🔊 Audio TTS généré: {len(result.audio_data)} bytes")
            print(f"⏱️ Latence TTS: {latence:.1f}ms")
            return result.audio_data
        else:
            print("❌ Échec synthèse TTS")
            return None
            
    except Exception as e:
        print(f"❌ Erreur TTS: {e}")
        return None

async def test_audio_playback(audio_data):
    """Test lecture audio"""
    if not audio_data:
        print("⚠️ Pas d'audio à jouer")
        return False
    
    print(f"\n🔈 TEST LECTURE AUDIO ({len(audio_data)} bytes)")
    
    try:
        # Initialiser audio manager
        audio_manager = AudioOutputManager()
        
        # Jouer l'audio
        await audio_manager.play_audio(audio_data)
        print("🔈 Audio joué avec succès !")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lecture audio: {e}")
        return False

async def test_pipeline_complet():
    """Test pipeline complet voix-à-voix"""
    print("\n🚀 TEST PIPELINE COMPLET VOIX-À-VOIX")
    print("=" * 60)
    
    debut_total = time.time()
    
    # Étape 1: Microphone → STT
    print("\n📍 ÉTAPE 1: CAPTURE VOCALE")
    texte_transcrit = await test_microphone_streaming()
    
    if not texte_transcrit:
        print("❌ Échec capture vocale - Arrêt du test")
        return False
    
    # Étape 2: STT → LLM
    print("\n📍 ÉTAPE 2: GÉNÉRATION RÉPONSE LLM")
    reponse_llm = await test_llm_response(texte_transcrit)
    
    if not reponse_llm:
        print("❌ Échec génération LLM - Arrêt du test")
        return False
    
    # Étape 3: LLM → TTS
    print("\n📍 ÉTAPE 3: SYNTHÈSE VOCALE")
    audio_tts = await test_tts_synthesis(reponse_llm)
    
    if not audio_tts:
        print("❌ Échec synthèse TTS - Arrêt du test")
        return False
    
    # Étape 4: TTS → Audio
    print("\n📍 ÉTAPE 4: LECTURE AUDIO")
    succes_audio = await test_audio_playback(audio_tts)
    
    fin_total = time.time()
    latence_totale = (fin_total - debut_total) * 1000
    
    # Résultats finaux
    print(f"\n🎊 RÉSULTATS PIPELINE COMPLET")
    print("=" * 50)
    print(f"📝 Input vocal: '{texte_transcrit}'")
    print(f"🤖 Réponse LLM: '{reponse_llm}'")
    print(f"🔊 Audio généré: {'✅ Oui' if audio_tts else '❌ Non'}")
    print(f"🔈 Audio joué: {'✅ Oui' if succes_audio else '❌ Non'}")
    print(f"⏱️ Latence totale: {latence_totale:.1f}ms")
    
    if latence_totale < 1200:
        print("🎯 OBJECTIF < 1200ms: ✅ ATTEINT")
    else:
        print("🎯 OBJECTIF < 1200ms: ❌ MANQUÉ")
    
    if succes_audio:
        print("\n🎊 PIPELINE VOIX-À-VOIX COMPLET RÉUSSI !")
        return True
    else:
        print("\n❌ Pipeline incomplet")
        return False

async def main():
    """Fonction principale"""
    print("🎤 TEST MICROPHONE DIRECT SUPERWHISPER V6")
    print("=" * 60)
    
    # Validation GPU obligatoire
    try:
        validate_rtx3090()
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    try:
        # Test pipeline complet
        succes = await test_pipeline_complet()
        
        if succes:
            print("\n🎊 TOUS LES TESTS RÉUSSIS !")
            return 0
        else:
            print("\n❌ Certains tests ont échoué")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu par l'utilisateur")
        return 1
    
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        return 1

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1) 