#!/usr/bin/env python3
"""
SuperWhisper V6 - Test Pipeline Voix-à-Voix avec Microphone Réel
==============================================================

Test complet : Parlez au microphone → STT → LLM → TTS → Réponse audio

Usage:
    python test_microphone_pipeline.py
    
Puis parlez dans le microphone et écoutez la réponse !
"""

import os
import sys
import time
import logging
import asyncio
import signal
import threading
from pathlib import Path

# Configuration GPU RTX 3090
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:1024")

# Ajout du projet au PATH
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MicrophonePipeline")

class MicrophonePipelineTest:
    """Test pipeline complet avec microphone"""
    
    def __init__(self):
        self.stt_manager = None
        self.llm_manager = None
        self.tts_manager = None
        self.running = False
        self.latest_transcription = ""
        self.transcription_ready = False
        
    def initialize_components(self):
        """Initialise tous les composants"""
        logger.info("🚀 Initialisation pipeline microphone SuperWhisper V6")
        
        # 1. STT Manager
        logger.info("🎤 Initialisation STT Manager...")
        from STT.unified_stt_manager import UnifiedSTTManager
        
        stt_config = {
            'timeout_per_minute': 10.0,
            'cache_size_mb': 200,
            'fallback_chain': ['prism_primary']
        }
        
        self.stt_manager = UnifiedSTTManager(stt_config)
        logger.info("✅ STT Manager initialisé")
        
        # 2. LLM Manager  
        logger.info("🧠 Initialisation LLM Manager...")
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        
        llm_config = {
            "model_name": "nous-hermes",
            "max_context_turns": 5,
            "timeout_seconds": 10,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        self.llm_manager = EnhancedLLMManager(llm_config)
        logger.info("✅ LLM Manager initialisé")
        
        # 3. TTS Manager
        logger.info("🔊 Initialisation TTS Manager...")
        from TTS.tts_manager import UnifiedTTSManager
        
        tts_config = {
            "cache": {
                "enabled": True,
                "max_size": 100,
                "ttl_seconds": 3600
            },
            "circuit_breaker": {
                "failure_threshold": 5,
                "reset_timeout_seconds": 30
            },
            "backends": {
                "silent_emergency": {
                    "enabled": True
                }
            },
            "default_handler": "silent_emergency"
        }
        
        self.tts_manager = UnifiedTTSManager(tts_config)
        logger.info("✅ TTS Manager initialisé")
        
        logger.info("🎉 Pipeline microphone prêt !")
        
    def record_audio_simple(self, duration_seconds=5):
        """Enregistrement audio simple avec sounddevice"""
        try:
            import sounddevice as sd
            import numpy as np
            
            logger.info(f"🎙️ Enregistrement {duration_seconds}s... Parlez maintenant !")
            
            # Enregistrement
            sample_rate = 16000
            audio_data = sd.rec(
                int(duration_seconds * sample_rate), 
                samplerate=sample_rate, 
                channels=1, 
                dtype=np.float32
            )
            sd.wait()  # Attendre la fin
            
            logger.info("✅ Enregistrement terminé")
            
            # Conversion en bytes pour STT
            audio_int16 = (audio_data * 32768).clip(-32768, 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            return audio_bytes, sample_rate
            
        except Exception as e:
            logger.error(f"❌ Erreur enregistrement: {e}")
            return None, 0
            
    def transcribe_audio(self, audio_bytes, sample_rate):
        """Transcription de l'audio"""
        try:
            start_time = time.time()
            
            # Sauvegarder temporairement l'audio pour STT
            import tempfile
            import wave
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Créer fichier WAV
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_bytes)
                
                # Transcription
                result = self.stt_manager.transcribe_file(temp_file.name)
                
                # Nettoyage
                os.unlink(temp_file.name)
                
                stt_time = time.time() - start_time
                logger.info(f"📝 Transcription ({stt_time:.1f}s): '{result}'")
                
                return result, stt_time
                
        except Exception as e:
            logger.error(f"❌ Erreur STT: {e}")
            return "", 0
            
    def generate_response(self, text):
        """Génération réponse LLM"""
        try:
            start_time = time.time()
            
            # Prompt conversationnel
            async def generate_async():
                return await self.llm_manager.generate_response(
                    user_input=text,
                    max_tokens=80,
                    temperature=0.7
                )
            
            response = asyncio.run(generate_async())
            llm_time = time.time() - start_time
            
            logger.info(f"🧠 Réponse LLM ({llm_time:.1f}s): '{response}'")
            return response, llm_time
            
        except Exception as e:
            logger.error(f"❌ Erreur LLM: {e}")
            return "Désolé, je n'ai pas compris.", 0
            
    def speak_response(self, text):
        """Lecture de la réponse"""
        try:
            start_time = time.time()
            
            # Synthèse vocale
            async def synthesize_async():
                return await self.tts_manager.synthesize(text=text)
            
            audio_data = asyncio.run(synthesize_async())
            tts_time = time.time() - start_time
            
            if audio_data and len(audio_data) > 0:
                # Sauvegarder et lire l'audio
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file.flush()
                    
                    # Lecture système
                    if sys.platform.startswith('win'):
                        os.system(f'start "" "{temp_file.name}"')
                    elif sys.platform.startswith('linux'):
                        os.system(f'aplay "{temp_file.name}" 2>/dev/null &')
                    elif sys.platform.startswith('darwin'):
                        os.system(f'afplay "{temp_file.name}" &')
                    
                    # Nettoyage différé
                    def cleanup():
                        time.sleep(3)
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
                    threading.Thread(target=cleanup, daemon=True).start()
                    
                logger.info(f"🔊 Audio joué ({tts_time:.1f}s)")
            else:
                logger.info(f"🔊 Réponse synthétisée ({tts_time:.1f}s) - mode silencieux")
                
            return tts_time
            
        except Exception as e:
            logger.error(f"❌ Erreur TTS: {e}")
            return 0
            
    def run_conversation_test(self):
        """Test conversation complète"""
        logger.info("\n" + "="*60)
        logger.info("🎯 TEST PIPELINE MICROPHONE DÉMARRÉ")
        logger.info("="*60)
        logger.info("🎤 Vous allez pouvoir parler dans le microphone")
        logger.info("🤖 Le système va transcrire, générer une réponse et la lire")
        logger.info("🛑 Ctrl+C pour arrêter")
        logger.info("="*60)
        
        conversation_count = 0
        
        try:
            while True:
                conversation_count += 1
                print(f"\n🎙️ Conversation #{conversation_count}")
                print("Appuyez sur Entrée quand vous êtes prêt à parler...")
                input()
                
                total_start = time.time()
                
                # 1. Enregistrement audio
                audio_bytes, sample_rate = self.record_audio_simple(duration_seconds=5)
                
                if not audio_bytes:
                    print("❌ Problème d'enregistrement, nouvelle tentative...")
                    continue
                
                # 2. Transcription STT
                transcription, stt_time = self.transcribe_audio(audio_bytes, sample_rate)
                
                if not transcription.strip():
                    print("⚠️ Aucune parole détectée, nouvelle tentative...")
                    continue
                
                # 3. Génération réponse LLM
                response, llm_time = self.generate_response(transcription)
                
                # 4. Synthèse et lecture TTS
                tts_time = self.speak_response(response)
                
                # 5. Statistiques
                total_time = time.time() - total_start
                
                print(f"\n📊 RÉSULTATS CONVERSATION #{conversation_count}")
                print(f"   📝 Vous avez dit: '{transcription}'")
                print(f"   🤖 Réponse: '{response}'")
                print(f"   ⏱️ Latences: STT {stt_time:.1f}s | LLM {llm_time:.1f}s | TTS {tts_time:.1f}s")
                print(f"   🎯 Temps total: {total_time:.1f}s")
                
                if total_time < 3.0:
                    print("   ✅ OBJECTIF <3s ATTEINT !")
                else:
                    print("   ⚠️ Latence > 3s")
                
                print("\nContinuer ? (Entrée = Oui, Ctrl+C = Stop)")
                
        except KeyboardInterrupt:
            logger.info(f"\n🛑 Test terminé après {conversation_count} conversations")
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")

def main():
    """Point d'entrée"""
    pipeline = MicrophonePipelineTest()
    
    try:
        # Initialisation
        pipeline.initialize_components()
        
        # Test conversation
        pipeline.run_conversation_test()
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()