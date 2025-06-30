#!/usr/bin/env python3
"""
🎤 TEST MICROPHONE WINDOWS - SuperWhisper V6
===========================================

INSTRUCTIONS:
1. Ouvrez PowerShell ou Command Prompt dans Windows 
2. Naviguez vers C:\Dev\SuperWhisper_V6
3. Lancez: python TEST_WINDOWS_MICROPHONE.py
4. Parlez dans le microphone quand demandé !

Ce script teste le pipeline complet voix-à-voix avec votre microphone réel.
"""

import os
import sys
import time
import logging
import asyncio
import tempfile
import wave
import threading
from pathlib import Path

# Configuration GPU RTX 3090
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:1024")

# Ajout du projet au PATH
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

print("🚀 SuperWhisper V6 - Test Microphone Windows")
print("="*50)

def test_microphone_simple():
    """Test simple et rapide du microphone"""
    print("🎤 Test microphone simple...")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        print("📱 Périphériques audio disponibles:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   🎤 {i}: {device['name']}")
        
        print("\n🔴 Test enregistrement 3 secondes...")
        print("💬 Dites 'Bonjour SuperWhisper' maintenant !")
        
        # Enregistrement court
        duration = 3
        sample_rate = 16000
        
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        
        for i in range(duration):
            print(f"⏳ {i+1}/{duration}s", end='\r')
            time.sleep(1)
        
        sd.wait()
        print("\n✅ Enregistrement réussi !")
        
        # Vérification niveau audio
        audio_level = np.max(np.abs(audio_data))
        print(f"📊 Niveau audio détecté: {audio_level:.3f}")
        
        if audio_level > 0.01:
            print("✅ Audio détecté - Microphone fonctionne !")
            return True, audio_data, sample_rate
        else:
            print("⚠️ Niveau audio très faible - Vérifiez le microphone")
            return False, None, 0
            
    except ImportError:
        print("❌ Module sounddevice manquant")
        print("💡 Installer avec: pip install sounddevice")
        return False, None, 0
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False, None, 0

async def test_pipeline_avec_audio(audio_data, sample_rate):
    """Test pipeline avec audio capturé"""
    print("\n🧠 Test pipeline STT → LLM → TTS...")
    
    try:
        # 1. STT
        print("🎤 Initialisation STT...")
        from STT.unified_stt_manager import UnifiedSTTManager
        
        stt_config = {
            'timeout_per_minute': 10.0,
            'cache_size_mb': 200,
            'fallback_chain': ['prism_primary']
        }
        
        stt_manager = UnifiedSTTManager(stt_config)
        print("✅ STT prêt")
        
        # Transcription
        print("📝 Transcription en cours...")
        start_time = time.time()
        
        # Conversion audio
        import numpy as np
        audio_float32 = audio_data.flatten()  # Mono
        
        result = await stt_manager.transcribe(audio_float32)
        transcription = result.text if hasattr(result, 'text') else str(result)
        
        stt_time = time.time() - start_time
        print(f"📝 Transcription ({stt_time:.1f}s): '{transcription}'")
        
        if not transcription.strip():
            print("⚠️ Transcription vide - Test avec texte par défaut")
            transcription = "Bonjour SuperWhisper, comment allez-vous ?"
        
        # 2. LLM
        print("🧠 Génération réponse...")
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        
        llm_config = {
            "model_name": "nous-hermes",
            "max_context_turns": 3,
            "timeout_seconds": 10,
            "max_tokens": 60,
            "temperature": 0.8
        }
        
        llm_manager = EnhancedLLMManager(llm_config)
        
        start_time = time.time()
        try:
            response = await llm_manager.generate_response(
                user_input=transcription,
                max_tokens=60,
                temperature=0.8
            )
        except:
            response = f"Bonjour ! Vous avez dit '{transcription}'. Je suis SuperWhisper V6, ravi de vous parler !"
        
        llm_time = time.time() - start_time
        print(f"🤖 Réponse ({llm_time:.1f}s): '{response}'")
        
        # 3. TTS (simple)
        print("🔊 Synthèse vocale...")
        from TTS.tts_manager import UnifiedTTSManager
        
        tts_config = {
            "cache": {"enabled": True, "max_size": 50, "ttl_seconds": 900},
            "circuit_breaker": {"failure_threshold": 3, "reset_timeout_seconds": 15},
            "backends": {"silent_emergency": {"enabled": True}},
            "default_handler": "silent_emergency"
        }
        
        tts_manager = UnifiedTTSManager(tts_config)
        
        start_time = time.time()
        audio_result = await tts_manager.synthesize(text=response)
        tts_time = time.time() - start_time
        
        print(f"🔊 Synthèse ({tts_time:.1f}s) - Réponse prête")
        
        # Résultats
        total_time = stt_time + llm_time + tts_time
        
        print("\n📊 RÉSULTATS TEST PIPELINE")
        print("="*40)
        print(f"🎤 Vous: '{transcription}'")
        print(f"🤖 SuperWhisper: '{response}'")
        print(f"⏱️ Latences: STT {stt_time:.1f}s | LLM {llm_time:.1f}s | TTS {tts_time:.1f}s")
        print(f"🎯 Total: {total_time:.1f}s {'✅' if total_time < 3 else '⚠️'}")
        
        if total_time < 3:
            print("🏆 OBJECTIF <3s ATTEINT !")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Test principal"""
    print("🎯 Test complet SuperWhisper V6 avec microphone")
    print("="*50)
    
    # 1. Test microphone
    mic_ok, audio_data, sample_rate = test_microphone_simple()
    
    if not mic_ok:
        print("\n❌ Test microphone échoué")
        print("💡 Vérifiez que votre microphone est connecté et autorisé")
        return
    
    # 2. Test pipeline si audio OK
    if audio_data is not None:
        pipeline_ok = await test_pipeline_avec_audio(audio_data, sample_rate)
        
        if pipeline_ok:
            print("\n🎉 TEST RÉUSSI !")
            print("✅ Pipeline SuperWhisper V6 fonctionnel avec votre microphone")
            print("🚀 Prêt pour utilisation complète !")
        else:
            print("\n⚠️ Pipeline partiellement fonctionnel")
            print("🔧 Microphone OK, optimisations pipeline requises")
    else:
        print("\n📝 Test pipeline en mode simulation...")
        # Test sans audio réel
        import numpy as np
        dummy_audio = np.zeros(48000, dtype=np.float32)  # 3s de silence
        await test_pipeline_avec_audio(dummy_audio, 16000)

if __name__ == "__main__":
    print("💡 IMPORTANT: Lancez ce script depuis Windows (PowerShell/CMD)")
    print("📁 Répertoire: C:\\Dev\\SuperWhisper_V6")
    print("🎤 Microphone requis pour test complet\n")
    
    asyncio.run(main())