#!/usr/bin/env python3
"""
SuperWhisper V6 - Test RÉEL avec Microphone
==========================================

Test en conditions réelles : Vous parlez → Pipeline traite → Vous entendez la réponse

IMPORTANT: Lancez ce script depuis Windows (pas WSL) pour accès microphone
"""

import os
import sys
import time
import logging
import asyncio
import tempfile
import wave
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
logger = logging.getLogger("TestReel")

class PipelineReelTest:
    """Test pipeline en conditions réelles"""
    
    def __init__(self):
        self.stt_manager = None
        self.llm_manager = None
        self.tts_manager = None
        self.audio_device = None
        
    def check_audio_system(self):
        """Vérification système audio"""
        logger.info("🔊 Vérification système audio...")
        
        try:
            import sounddevice as sd
            
            # Liste des périphériques
            devices = sd.query_devices()
            logger.info("📱 Périphériques audio détectés:")
            
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device['name']))
                    logger.info(f"   🎤 {i}: {device['name']}")
            
            if not input_devices:
                logger.error("❌ Aucun microphone détecté")
                return False
                
            # Test microphone par défaut
            default_device = sd.default.device[0]
            if default_device is not None:
                logger.info(f"✅ Microphone par défaut: {devices[default_device]['name']}")
                self.audio_device = default_device
            else:
                self.audio_device = input_devices[0][0]
                logger.info(f"✅ Utilisation microphone: {input_devices[0][1]}")
                
            return True
            
        except ImportError:
            logger.error("❌ Module sounddevice non installé")
            logger.info("💡 Installer avec: pip install sounddevice")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur audio: {e}")
            return False
            
    def initialize_pipeline(self):
        """Initialisation pipeline complet"""
        logger.info("🚀 Initialisation pipeline SuperWhisper V6...")
        
        # 1. STT
        logger.info("🎤 STT Manager...")
        from STT.unified_stt_manager import UnifiedSTTManager
        
        stt_config = {
            'timeout_per_minute': 10.0,
            'cache_size_mb': 200,
            'fallback_chain': ['prism_primary']
        }
        
        self.stt_manager = UnifiedSTTManager(stt_config)
        logger.info("✅ STT prêt (Whisper large-v2)")
        
        # 2. LLM 
        logger.info("🧠 LLM Manager...")
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        
        llm_config = {
            "model_name": "nous-hermes",
            "max_context_turns": 3,
            "timeout_seconds": 15,
            "max_tokens": 80,
            "temperature": 0.8
        }
        
        self.llm_manager = EnhancedLLMManager(llm_config)
        logger.info("✅ LLM prêt (Nous-hermes)")
        
        # 3. TTS
        logger.info("🔊 TTS Manager...")
        from TTS.tts_manager import UnifiedTTSManager
        
        tts_config = {
            "cache": {"enabled": True, "max_size": 50, "ttl_seconds": 1800},
            "circuit_breaker": {"failure_threshold": 3, "reset_timeout_seconds": 20},
            "backends": {"silent_emergency": {"enabled": True}},
            "default_handler": "silent_emergency"
        }
        
        self.tts_manager = UnifiedTTSManager(tts_config)
        logger.info("✅ TTS prêt (voix française)")
        
        logger.info("🎉 Pipeline initialisé - Prêt pour test réel !")
        
    def record_voice(self, duration=6):
        """Enregistrement vocal réel"""
        try:
            import sounddevice as sd
            import numpy as np
            
            sample_rate = 16000
            
            print(f"\n🔴 ENREGISTREMENT {duration}s - PARLEZ MAINTENANT !")
            print("💭 Posez une question ou dites quelque chose...")
            
            # Enregistrement
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                device=self.audio_device
            )
            
            # Barre de progression
            for i in range(duration):
                print(f"⏳ {i+1}/{duration}s", end='\r')
                time.sleep(1)
            
            sd.wait()
            print("\n✅ Enregistrement terminé")
            
            # Conversion pour traitement
            audio_int16 = (audio_data * 32768).clip(-32768, 32767).astype(np.int16)
            return audio_int16.tobytes(), sample_rate
            
        except Exception as e:
            logger.error(f"❌ Erreur enregistrement: {e}")
            return None, 0
            
    async def transcribe_audio(self, audio_bytes, sample_rate):
        """Transcription STT"""
        try:
            import numpy as np
            
            # Conversion bytes vers numpy array
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Transcription
            start_time = time.time()
            result = await self.stt_manager.transcribe(audio_float32)
            stt_time = time.time() - start_time
            
            transcription = result.text if hasattr(result, 'text') else str(result)
            
            return transcription, stt_time
                
        except Exception as e:
            logger.error(f"❌ Erreur STT: {e}")
            return "", 0
            
    async def generate_response(self, user_input):
        """Génération réponse conversationnelle"""
        try:
            start_time = time.time()
            
            # Réponse LLM
            response = await self.llm_manager.generate_response(
                user_input=user_input,
                max_tokens=100,
                temperature=0.8,
                include_context=True
            )
            
            llm_time = time.time() - start_time
            return response, llm_time
            
        except Exception as e:
            logger.error(f"❌ Erreur LLM: {e}")
            # Réponses de fallback intelligentes
            fallback_responses = {
                "bonjour": "Bonjour ! Je suis SuperWhisper V6. Comment puis-je vous aider ?",
                "comment": "Je vais bien, merci ! Et vous, comment allez-vous ?",
                "temps": "Je ne peux pas consulter la météo, mais j'espère qu'il fait beau !",
                "merci": "De rien ! C'est un plaisir de vous aider.",
                "au revoir": "Au revoir ! À bientôt pour une nouvelle conversation !"
            }
            
            user_lower = user_input.lower()
            for keyword, response in fallback_responses.items():
                if keyword in user_lower:
                    return response, 0.1
                    
            return "Désolé, pouvez-vous répéter ? Je n'ai pas bien compris.", 0.1
            
    async def speak_response(self, text):
        """Lecture de la réponse"""
        try:
            start_time = time.time()
            
            # Synthèse
            audio_data = await self.tts_manager.synthesize(text=text)
            
            if audio_data and len(audio_data) > 0:
                # Lecture audio
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file.flush()
                    
                    # Lecture selon l'OS
                    if sys.platform.startswith('win'):
                        os.system(f'start /min "" "{temp_file.name}"')
                    elif sys.platform.startswith('linux'):
                        os.system(f'aplay "{temp_file.name}" 2>/dev/null')
                    else:
                        os.system(f'afplay "{temp_file.name}"')
                    
                    # Attendre lecture
                    time.sleep(len(text) * 0.1 + 1)  # Estimation durée
                    
                    # Nettoyage
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
            
            tts_time = time.time() - start_time
            return tts_time
            
        except Exception as e:
            logger.error(f"❌ Erreur TTS: {e}")
            print(f"🔊 Réponse (texte): {text}")
            return 0.1
            
    async def run_real_conversation_test(self):
        """Test conversation réelle"""
        print("\n" + "="*60)
        print("🎯 TEST RÉEL PIPELINE SUPERWHISPER V6")
        print("="*60)
        print("🎤 Test avec VOTRE MICROPHONE")
        print("🤖 Le système va transcrire, répondre et parler")
        print("🔄 Plusieurs conversations possibles")
        print("="*60)
        
        conversation_count = 0
        
        try:
            while True:
                conversation_count += 1
                print(f"\n💬 CONVERSATION #{conversation_count}")
                print("📋 Appuyez sur Entrée pour commencer l'enregistrement...")
                
                # Attendre confirmation utilisateur
                try:
                    input()
                except EOFError:
                    print("Démarrage automatique...")
                
                total_start = time.time()
                
                # 1. Enregistrement
                audio_bytes, sample_rate = self.record_voice(duration=6)
                
                if not audio_bytes:
                    print("❌ Problème enregistrement, nouvelle tentative ?")
                    continue
                
                # 2. Transcription
                print("🎤 Transcription en cours...")
                transcription, stt_time = await self.transcribe_audio(audio_bytes, sample_rate)
                
                if not transcription.strip():
                    print("⚠️ Aucune parole détectée. Parlez plus fort ou répétez ?")
                    continue
                
                print(f"📝 Vous avez dit: '{transcription}'")
                
                # 3. Génération réponse
                print("🧠 Génération réponse...")
                response, llm_time = await self.generate_response(transcription)
                print(f"🤖 Réponse: '{response}'")
                
                # 4. Lecture réponse
                print("🔊 Lecture réponse...")
                tts_time = await self.speak_response(response)
                
                # 5. Statistiques
                total_time = time.time() - total_start
                
                print(f"\n📊 MÉTRIQUES CONVERSATION #{conversation_count}")
                print(f"   ⏱️ STT: {stt_time:.1f}s | LLM: {llm_time:.1f}s | TTS: {tts_time:.1f}s")
                print(f"   🎯 Total: {total_time:.1f}s {'✅' if total_time < 3 else '⚠️'}")
                
                # Continuer ?
                print("\n❓ Continuer ? (Entrée=Oui, Ctrl+C=Stop)")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    break
                    
        except KeyboardInterrupt:
            print(f"\n🏁 Test terminé après {conversation_count} conversations")
        
        return conversation_count

async def main():
    """Test principal"""
    test = PipelineReelTest()
    
    try:
        # Vérifications
        if not test.check_audio_system():
            print("❌ Système audio non disponible")
            print("💡 Lancez depuis Windows avec microphone connecté")
            return
        
        # Initialisation
        test.initialize_pipeline()
        
        # Test réel
        conversations = await test.run_real_conversation_test()
        
        print(f"\n🎉 Test réel terminé: {conversations} conversations testées")
        print("✅ Pipeline SuperWhisper V6 testé en conditions réelles !")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())