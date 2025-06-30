#!/usr/bin/env python3
"""
SuperWhisper V6 - Démonstration Pipeline Voix-à-Voix
====================================================

Pipeline complet : Microphone → STT → LLM → TTS → Haut-parleurs
Composants validés intégrés pour conversation temps réel.

Usage:
    python demo_pipeline_voix_a_voix.py
    
Puis parlez dans le microphone et écoutez la réponse !
"""

import os
import sys
import time
import logging
import signal
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
logger = logging.getLogger("VoiceToVoice")

class VoiceToVoicePipeline:
    """Pipeline voix-à-voix intégré pour SuperWhisper V6"""
    
    def __init__(self):
        self.stt_manager = None
        self.streaming_mic = None
        self.llm_manager = None
        self.tts_manager = None
        self.running = False
        
        # Statistiques
        self.conversation_count = 0
        self.total_latency = 0
        
    def initialize_components(self):
        """Initialise tous les composants du pipeline"""
        logger.info("🚀 Initialisation pipeline voix-à-voix SuperWhisper V6")
        
        # 1. STT avec StreamingMicrophoneManager
        logger.info("🎤 Initialisation STT Manager...")
        from STT.unified_stt_manager import UnifiedSTTManager
        from STT.streaming_microphone_manager import StreamingMicrophoneManager
        
        stt_config = {
            'timeout_per_minute': 10.0,
            'cache_size_mb': 200,
            'fallback_chain': ['prism_primary']
        }
        
        self.stt_manager = UnifiedSTTManager(stt_config)
        self.streaming_mic = StreamingMicrophoneManager(self.stt_manager)
        logger.info("✅ STT initialisé avec StreamingMicrophoneManager")
        
        # 2. LLM Manager
        logger.info("🧠 Initialisation LLM Manager...")
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        
        llm_config = {
            "model_name": "nous-hermes",
            "max_context_turns": 5,
            "timeout_seconds": 10,
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        self.llm_manager = EnhancedLLMManager(llm_config)
        logger.info("✅ LLM Manager initialisé (Nous-hermes)")
        
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
                "piper_native": {
                    "enabled": True,
                    "model_path": "/opt/piper/models/fr_FR-upmc-medium.onnx",
                    "speaker_id": 0
                },
                "sapi_french": {
                    "enabled": True,
                    "speaker_id": 0
                }
            },
            "default_handler": "piper_native"
        }
        
        self.tts_manager = UnifiedTTSManager(tts_config)
        logger.info("✅ TTS Manager initialisé (voix française)")
        
        logger.info("🎉 Pipeline voix-à-voix prêt !")
        
    def process_voice_input(self, timeout_seconds=10):
        """Capture et traite une entrée vocale"""
        logger.info(f"🎙️ Écoute pendant {timeout_seconds}s... Parlez maintenant !")
        
        start_time = time.time()
        transcription = ""
        
        # Démarrer le streaming microphone
        self.streaming_mic.start_streaming()
        
        try:
            # Attendre la transcription avec timeout
            elapsed = 0
            while elapsed < timeout_seconds:
                if hasattr(self.streaming_mic, 'get_latest_transcription'):
                    latest = self.streaming_mic.get_latest_transcription()
                    if latest and latest.strip():
                        transcription = latest
                        break
                
                time.sleep(0.1)
                elapsed = time.time() - start_time
                
        finally:
            self.streaming_mic.stop_streaming()
            
        stt_time = time.time() - start_time
        
        if not transcription.strip():
            logger.warning("⚠️ Aucune transcription détectée")
            return None, 0
            
        logger.info(f"📝 Transcription ({stt_time:.1f}s): '{transcription}'")
        return transcription, stt_time
        
    def generate_llm_response(self, text):
        """Génère une réponse LLM"""
        start_time = time.time()
        
        try:
            # Prompt conversationnel simple
            conversation_prompt = f"""Tu es un assistant vocal français amical et concis.
Réponds de manière naturelle et courte (maximum 2 phrases) à cette question ou remarque :

"{text}"

Réponse naturelle:"""
            
            response = self.llm_manager.generate_response(
                prompt=conversation_prompt,
                max_tokens=100,
                temperature=0.7
            )
            
            llm_time = time.time() - start_time
            logger.info(f"🧠 Réponse LLM ({llm_time:.1f}s): '{response}'")
            return response, llm_time
            
        except Exception as e:
            logger.error(f"❌ Erreur LLM: {e}")
            return "Désolé, je n'ai pas pu traiter votre demande.", 0
            
    def speak_response(self, text):
        """Synthétise et joue la réponse"""
        start_time = time.time()
        
        try:
            import asyncio
            
            # Génération audio asynchrone
            async def synthesize_async():
                audio_data = await self.tts_manager.synthesize(
                    text=text,
                    voice="french_female"
                )
                return audio_data
            
            # Exécuter la synthèse
            audio_data = asyncio.run(synthesize_async())
            
            if audio_data:
                # Sauvegarder temporairement l'audio
                import tempfile
                import wave
                
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_file.write(audio_data)
                temp_file.close()
                
                # Lecture avec système par défaut
                if sys.platform.startswith('win'):
                    os.system(f'start "" "{temp_file.name}"')
                elif sys.platform.startswith('linux'):
                    os.system(f'aplay "{temp_file.name}" 2>/dev/null &')
                elif sys.platform.startswith('darwin'):
                    os.system(f'afplay "{temp_file.name}" &')
                    
                # Nettoyage après délai
                import threading
                def cleanup_temp():
                    time.sleep(5)
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                threading.Thread(target=cleanup_temp, daemon=True).start()
                        
                tts_time = time.time() - start_time
                logger.info(f"🔊 Audio joué ({tts_time:.1f}s)")
                return tts_time
                    
        except Exception as e:
            logger.error(f"❌ Erreur TTS: {e}")
            
        return 0
        
    def run_conversation_loop(self):
        """Boucle principale de conversation"""
        logger.info("\n" + "="*60)
        logger.info("🎯 PIPELINE VOIX-À-VOIX DÉMARRÉ")
        logger.info("="*60)
        logger.info("💬 Parlez dans le microphone pour démarrer une conversation")
        logger.info("🛑 Ctrl+C pour arrêter")
        logger.info("="*60)
        
        self.running = True
        
        try:
            while self.running:
                print(f"\n🎙️ Conversation #{self.conversation_count + 1} - Parlez maintenant...")
                
                # 1. Capture vocale
                conversation_start = time.time()
                transcription, stt_time = self.process_voice_input(timeout_seconds=8)
                
                if not transcription:
                    print("⏳ Aucune parole détectée, nouvelle tentative...")
                    continue
                    
                # 2. Génération réponse LLM
                llm_response, llm_time = self.generate_llm_response(transcription)
                
                # 3. Synthèse et lecture
                tts_time = self.speak_response(llm_response)
                
                # 4. Statistiques
                total_time = time.time() - conversation_start
                self.conversation_count += 1
                self.total_latency += total_time
                
                print(f"⏱️ Latences: STT {stt_time:.1f}s | LLM {llm_time:.1f}s | TTS {tts_time:.1f}s | Total {total_time:.1f}s")
                print(f"📊 Conversations: {self.conversation_count} | Latence moyenne: {self.total_latency/self.conversation_count:.1f}s")
                
                # Pause avant la prochaine conversation
                time.sleep(2)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Arrêt demandé par l'utilisateur")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Nettoyage des ressources"""
        self.running = False
        if self.streaming_mic:
            try:
                self.streaming_mic.stop_streaming()
            except:
                pass
        logger.info("✅ Pipeline arrêté proprement")

def main():
    """Point d'entrée principal"""
    pipeline = VoiceToVoicePipeline()
    
    try:
        # Initialisation
        pipeline.initialize_components()
        
        # Démarrage conversation
        pipeline.run_conversation_loop()
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main()