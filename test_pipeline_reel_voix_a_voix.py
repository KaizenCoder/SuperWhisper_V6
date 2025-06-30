#!/usr/bin/env python3
"""
Test Pipeline Réel Voix-à-Voix SuperWhisper V6
Microphone → STT → LLM → TTS → Audio en temps réel
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
logger = logging.getLogger("VoixAVoix")

class PipelineVoixAVoix:
    """Pipeline voix-à-voix temps réel avec microphone streaming"""
    
    def __init__(self):
        self.stt_manager = None
        self.llm_manager = None
        self.tts_manager = None
        self.running = False
        self.conversation_count = 0
        
    def initialize_components(self):
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
            
            # 2. LLM Manager 
            from LLM.llm_manager_enhanced import EnhancedLLMManager
            llm_config = {
                'model': 'nous-hermes',
                'base_url': 'http://127.0.0.1:11434/v1',
                'timeout': 30.0
            }
            self.llm_manager = EnhancedLLMManager(llm_config)
            logger.info("✅ LLM Manager initialisé")
            
            # 3. TTS Manager (validé)
            from TTS.tts_manager import UnifiedTTSManager
            import yaml
            with open('config/tts.yaml', 'r') as f:
                tts_config = yaml.safe_load(f)
            self.tts_manager = UnifiedTTSManager(tts_config)
            logger.info("✅ TTS Manager initialisé")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation: {e}")
            return False
    
    def capture_audio_from_microphone(self, duration_seconds=5):
        """Capture audio depuis le microphone"""
        logger.info(f"🎤 Capture audio pendant {duration_seconds}s...")
        logger.info("💬 Parlez maintenant dans le microphone !")
        
        try:
            import sounddevice as sd
            import numpy as np
            
            # Configuration audio
            sample_rate = 16000
            channels = 1
            
            # Capture audio
            logger.info("🔴 Enregistrement en cours...")
            audio_data = sd.rec(
                int(duration_seconds * sample_rate), 
                samplerate=sample_rate, 
                channels=channels,
                dtype=np.float32
            )
            sd.wait()  # Attendre la fin de l'enregistrement
            
            logger.info("✅ Enregistrement terminé")
            
            # Convertir en bytes pour STT
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"❌ Erreur capture audio: {e}")
            return None
    
    async def process_stt(self, audio_bytes):
        """Traite l'audio avec STT"""
        logger.info("🎤 Traitement STT...")
        start_time = time.perf_counter()
        
        try:
            # Transcription audio avec la méthode correcte
            result = await self.stt_manager.transcribe_pcm(audio_bytes, 16000)
            
            stt_time = (time.perf_counter() - start_time) * 1000
            
            if result and result.strip():
                logger.info(f"✅ STT réussi ({stt_time:.1f}ms): '{result}'")
                return result, stt_time
            else:
                logger.warning("⚠️ STT vide - aucun texte détecté")
                return None, stt_time
                
        except Exception as e:
            logger.error(f"❌ Erreur STT: {e}")
            return None, 0
    
    async def process_llm(self, user_text):
        """Traite le texte avec LLM"""
        logger.info("🧠 Traitement LLM...")
        start_time = time.perf_counter()
        
        try:
            # Génération réponse LLM
            response = await self.llm_manager.generate_response(
                user_input=user_text,
                max_tokens=100
            )
            
            llm_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"✅ LLM réussi ({llm_time:.1f}ms): '{response}'")
            return response, llm_time
            
        except Exception as e:
            logger.error(f"❌ Erreur LLM: {e}")
            fallback_response = "Je suis désolé, je n'ai pas pu traiter votre demande."
            return fallback_response, 0
    
    async def process_tts(self, response_text):
        """Traite le texte avec TTS"""
        logger.info("🔊 Synthèse TTS...")
        start_time = time.perf_counter()
        
        try:
            # Synthèse vocale
            tts_result = await self.tts_manager.synthesize(text=response_text)
            
            tts_time = (time.perf_counter() - start_time) * 1000
            
            if tts_result.success and tts_result.audio_data:
                # Sauvegarder et jouer l'audio
                conversation_file = f"conversation_{self.conversation_count}.wav"
                with open(conversation_file, 'wb') as f:
                    f.write(tts_result.audio_data)
                
                logger.info(f"✅ TTS réussi ({tts_time:.1f}ms)")
                logger.info(f"📁 Audio sauvegardé: {conversation_file}")
                logger.info(f"🎛️ Backend: {tts_result.backend_used}")
                
                # Tentative de lecture automatique
                try:
                    import subprocess
                    logger.info("🔊 Lecture audio...")
                    # Pour Windows WSL, essayer avec explorer.exe
                    subprocess.Popen([
                        "cmd.exe", "/c", "start", "", 
                        os.path.abspath(conversation_file).replace('/mnt/c', 'C:')
                    ], shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    logger.info(f"🎧 Lisez manuellement: {conversation_file}")
                
                return True, tts_time, conversation_file
            else:
                logger.error(f"❌ TTS échoué: {tts_result.error}")
                return False, tts_time, None
                
        except Exception as e:
            logger.error(f"❌ Erreur TTS: {e}")
            return False, 0, None
    
    async def run_conversation_cycle(self):
        """Exécute un cycle complet de conversation voix-à-voix"""
        self.conversation_count += 1
        
        logger.info("\n" + "="*60)
        logger.info(f"🎯 CONVERSATION #{self.conversation_count}")
        logger.info("="*60)
        
        total_start = time.perf_counter()
        
        # 1. Capture microphone
        audio_bytes = self.capture_audio_from_microphone(duration_seconds=5)
        if not audio_bytes:
            logger.error("❌ Échec capture audio")
            return False
        
        # 2. STT
        user_text, stt_time = await self.process_stt(audio_bytes)
        if not user_text:
            logger.warning("⚠️ Aucun texte détecté - cycle abandonné")
            return False
        
        # 3. LLM
        llm_response, llm_time = await self.process_llm(user_text)
        
        # 4. TTS
        tts_success, tts_time, audio_file = await self.process_tts(llm_response)
        
        # 5. Résultats
        total_time = (time.perf_counter() - total_start) * 1000
        
        logger.info("\n📊 RÉSULTATS CONVERSATION")
        logger.info("-" * 40)
        logger.info(f"👤 Utilisateur: '{user_text}'")
        logger.info(f"🤖 Assistant: '{llm_response}'")
        logger.info(f"⏱️ STT: {stt_time:.1f}ms")
        logger.info(f"⏱️ LLM: {llm_time:.1f}ms") 
        logger.info(f"⏱️ TTS: {tts_time:.1f}ms")
        logger.info(f"⏱️ TOTAL: {total_time:.1f}ms")
        
        if audio_file:
            logger.info(f"🎧 Audio: {audio_file}")
        
        return tts_success
    
    async def run_interactive_session(self):
        """Session interactive continue"""
        logger.info("\n🚀 SESSION VOIX-À-VOIX INTERACTIVE")
        logger.info("="*60)
        logger.info("💬 Parlez dans le microphone pour commencer")
        logger.info("🛑 Ctrl+C pour arrêter")
        logger.info("="*60)
        
        self.running = True
        
        try:
            while self.running:
                success = await self.run_conversation_cycle()
                
                if success:
                    logger.info("\n✅ Conversation réussie !")
                else:
                    logger.warning("\n⚠️ Conversation partiellement réussie")
                
                # Pause avant la prochaine conversation
                logger.info("\n⏳ Pause 3s avant la prochaine conversation...")
                await asyncio.sleep(3)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Session arrêtée par l'utilisateur")
        except Exception as e:
            logger.error(f"\n❌ Erreur session: {e}")
        finally:
            self.running = False

def main():
    """Point d'entrée principal"""
    logger.info("🚀 SuperWhisper V6 - Test Pipeline Réel Voix-à-Voix")
    
    # Gestionnaire de signal pour arrêt propre
    def signal_handler(signum, frame):
        logger.info("\n🛑 Arrêt demandé...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialiser le pipeline
        pipeline = PipelineVoixAVoix()
        
        if not pipeline.initialize_components():
            logger.error("❌ Échec initialisation - arrêt")
            return
        
        # Lancer la session interactive
        asyncio.run(pipeline.run_interactive_session())
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")

if __name__ == "__main__":
    main()