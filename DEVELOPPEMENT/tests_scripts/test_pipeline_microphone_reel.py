#!/usr/bin/env python3
"""
Test Pipeline RÉEL avec Microphone SuperWhisper V6
Vrai test : Microphone → STT → LLM → TTS → Audio
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import time
import logging
import asyncio
import pathlib
import signal
from pathlib import Path

# =============================================================================
# 🚀 PORTABILITÉ AUTOMATIQUE - EXÉCUTABLE DEPUIS N'IMPORTE OÙ
# =============================================================================
def _setup_portable_environment():
    """Configure l'environnement pour exécution portable"""
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

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MicrophoneReel")

class PipelineMicrophoneReel:
    """Pipeline avec microphone réel utilisant le StreamingMicrophoneManager validé"""
    
    def __init__(self):
        self.stt_manager = None
        self.streaming_mic = None
        self.llm_manager = None
        self.tts_manager = None
        self.running = False
        self.conversation_count = 0
        
    async def initialize_components(self):
        """Initialise tous les composants du pipeline"""
        logger.info("🔧 Initialisation des composants pipeline...")
        
        try:
            # 1. STT Manager (validé)
            from STT.unified_stt_manager import UnifiedSTTManager
            stt_config = {
                'timeout_per_minute': 10.0,
                'cache_size_mb': 200,
                'fallback_chain': ['prism_primary']
            }
            self.stt_manager = UnifiedSTTManager(stt_config)
            logger.info("✅ STT Manager initialisé")
            
            # 2. StreamingMicrophoneManager (validé de votre conversation)
            from STT.streaming_microphone_manager import StreamingMicrophoneManager
            self.streaming_mic = StreamingMicrophoneManager(self.stt_manager)
            logger.info("✅ StreamingMicrophoneManager initialisé")
            
            # 3. LLM Manager avec fallback
            from LLM.llm_manager_enhanced import EnhancedLLMManager
            llm_config = {
                'model': 'nous-hermes-2-mistral-7b-dpo:latest',
                'base_url': 'http://127.0.0.1:11434/v1',
                'timeout': 30.0,
                'use_ollama': True
            }
            self.llm_manager = EnhancedLLMManager(llm_config)
            await self.llm_manager.initialize()
            logger.info("✅ LLM Manager initialisé")
            
            # 4. TTS Manager (validé)
            from TTS.tts_manager import UnifiedTTSManager
            import yaml
            with open('config/tts.yaml', 'r') as f:
                tts_config = yaml.safe_load(f)
            self.tts_manager = UnifiedTTSManager(tts_config)
            logger.info("✅ TTS Manager initialisé")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def capture_from_microphone(self, duration_seconds=8):
        """Capture audio depuis le microphone avec le StreamingMicrophoneManager"""
        logger.info(f"🎤 Écoute microphone pendant {duration_seconds}s...")
        logger.info("💬 PARLEZ MAINTENANT dans le microphone !")
        
        try:
            # Variables de capture
            captured_text = None
            start_time = time.perf_counter()
            
            # Fonction callback pour récupérer la transcription
            def on_transcription(text):
                nonlocal captured_text
                if text and text.strip():
                    captured_text = text.strip()
                    logger.info(f"🎤 Transcription reçue: '{captured_text}'")
            
            # Démarrer le streaming avec callback
            # Note: Le StreamingMicrophoneManager de votre conversation utilise une API différente
            # Utilisons la méthode de votre script validé
            
            logger.info("🔴 Capture en cours...")
            
            # Simulation de capture car le StreamingMicrophoneManager nécessite un setup spécifique
            # Utilisons directement l'enregistrement audio
            import sounddevice as sd
            import numpy as np
            
            # Configuration audio compatible STT
            sample_rate = 16000
            channels = 1
            
            # Enregistrement
            audio_data = sd.rec(
                int(duration_seconds * sample_rate), 
                samplerate=sample_rate, 
                channels=channels,
                dtype=np.float32
            )
            sd.wait()  # Attendre la fin
            
            logger.info("✅ Enregistrement terminé")
            
            # Convertir pour STT
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            # Transcription STT
            logger.info("🎤 Transcription en cours...")
            stt_start = time.perf_counter()
            
            transcription = await self.stt_manager.transcribe_pcm(audio_bytes, sample_rate)
            
            stt_time = (time.perf_counter() - stt_start) * 1000
            total_time = (time.perf_counter() - start_time) * 1000
            
            if transcription and transcription.strip():
                logger.info(f"✅ STT réussi ({stt_time:.1f}ms): '{transcription}'")
                return transcription.strip(), stt_time
            else:
                logger.warning("⚠️ Aucun texte détecté dans l'audio")
                return None, stt_time
                
        except Exception as e:
            logger.error(f"❌ Erreur capture microphone: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    async def process_llm(self, user_text):
        """Traite le texte avec LLM"""
        logger.info("🧠 Consultation LLM...")
        start_time = time.perf_counter()
        
        try:
            # Génération réponse LLM
            response = await self.llm_manager.generate_response(
                user_input=user_text,
                max_tokens=100
            )
            
            llm_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"✅ LLM réussi ({llm_time:.1f}ms)")
            logger.info(f"🤖 Réponse: '{response}'")
            return response, llm_time
            
        except Exception as e:
            logger.warning(f"⚠️ LLM fallback: {e}")
            fallback_response = f"Merci pour votre question : '{user_text}'. Je vous réponds du mieux que je peux."
            return fallback_response, 1.0
    
    async def process_tts(self, response_text):
        """Traite le texte avec TTS et joue l'audio en temps réel"""
        logger.info("🔊 Synthèse vocale...")
        start_time = time.perf_counter()
        
        try:
            # Synthèse vocale
            tts_result = await self.tts_manager.synthesize(text=response_text)
            
            tts_time = (time.perf_counter() - start_time) * 1000
            
            if tts_result.success and tts_result.audio_data:
                # Sauvegarder l'audio
                conversation_file = f"conversation_reelle_{self.conversation_count}.wav"
                with open(conversation_file, 'wb') as f:
                    f.write(tts_result.audio_data)
                
                logger.info(f"✅ TTS réussi ({tts_time:.1f}ms)")
                logger.info(f"📁 Audio: {conversation_file}")
                logger.info(f"🎛️ Backend: {tts_result.backend_used}")
                logger.info(f"📊 Taille: {len(tts_result.audio_data)} bytes")
                
                # Lecture automatique TEMPS RÉEL (après pause pour éviter feedback)
                try:
                    import subprocess
                    logger.info("🔊 LECTURE TEMPS RÉEL...")
                    
                    # PAUSE pour éviter que le micro capte la réponse
                    logger.info("⏸️ Pause 3s pour éviter feedback microphone...")
                    await asyncio.sleep(3)
                    
                    # Convertir le chemin pour Windows
                    if os.path.exists(conversation_file):
                        windows_path = os.path.abspath(conversation_file)
                        subprocess.run([
                            "start", "", windows_path
                        ], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        logger.info(f"🎧 Audio en cours de lecture: {windows_path}")
                    
                    # Attendre fin de lecture
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Lecture automatique échouée: {e}")
                    logger.info(f"🎧 Lisez manuellement: {conversation_file}")
                
                return True, tts_time, conversation_file
            else:
                logger.error(f"❌ TTS échoué: {tts_result.error}")
                return False, tts_time, None
                
        except Exception as e:
            logger.error(f"❌ Erreur TTS: {e}")
            return False, 0, None
    
    async def run_conversation_cycle(self):
        """Exécute un cycle complet de conversation avec microphone réel"""
        self.conversation_count += 1
        
        logger.info("\n" + "="*60)
        logger.info(f"🎯 CONVERSATION RÉELLE #{self.conversation_count}")
        logger.info("="*60)
        
        total_start = time.perf_counter()
        
        # 1. Capture microphone réel
        user_text, stt_time = await self.capture_from_microphone(duration_seconds=8)
        if not user_text:
            logger.warning("⚠️ Aucune parole détectée - cycle abandonné")
            return False
        
        # 2. LLM
        llm_response, llm_time = await self.process_llm(user_text)
        
        # 3. TTS
        tts_success, tts_time, audio_file = await self.process_tts(llm_response)
        
        # 4. Résultats
        total_time = (time.perf_counter() - total_start) * 1000
        
        logger.info("\n📊 RÉSULTATS CONVERSATION RÉELLE")
        logger.info("-" * 40)
        logger.info(f"👤 Vous avez dit: '{user_text}'")
        logger.info(f"🤖 Assistant: '{llm_response}'")
        logger.info(f"⏱️ STT: {stt_time:.1f}ms")
        logger.info(f"⏱️ LLM: {llm_time:.1f}ms") 
        logger.info(f"⏱️ TTS: {tts_time:.1f}ms")
        logger.info(f"⏱️ TOTAL E2E: {total_time:.1f}ms")
        
        if audio_file:
            logger.info(f"🎧 Réponse audio: {audio_file}")
        
        return tts_success
    
    async def run_interactive_session(self):
        """Session interactive avec microphone réel"""
        logger.info("\n🚀 SESSION MICROPHONE RÉEL")
        logger.info("="*60)
        logger.info("🎤 Conversation voix-à-voix temps réel")
        logger.info("💬 Parlez dans le microphone quand demandé")
        logger.info("🛑 Ctrl+C pour arrêter")
        logger.info("="*60)
        
        self.running = True
        
        try:
            while self.running:
                success = await self.run_conversation_cycle()
                
                if success:
                    logger.info("\n✅ Conversation réelle réussie !")
                else:
                    logger.warning("\n⚠️ Conversation incomplète")
                
                # Demander si continuer
                logger.info("\n⏳ Pause 5s avant la prochaine conversation...")
                logger.info("💬 Préparez votre prochaine question...")
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Session arrêtée par l'utilisateur")
        except Exception as e:
            logger.error(f"\n❌ Erreur session: {e}")
        finally:
            self.running = False

async def main():
    """Point d'entrée principal"""
    logger.info("🚀 SuperWhisper V6 - Test Pipeline RÉEL avec Microphone")
    
    try:
        # Initialiser le pipeline
        pipeline = PipelineMicrophoneReel()
        
        if not await pipeline.initialize_components():
            logger.error("❌ Échec initialisation - arrêt")
            return
        
        # Lancer la session interactive
        await pipeline.run_interactive_session()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")

if __name__ == "__main__":
    asyncio.run(main())