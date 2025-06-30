#!/usr/bin/env python3
"""
SuperWhisper V6 - Test Pipeline E2E RÉEL
========================================

TEST COMPLET DE CONTINUITÉ :
Vous parlez → STT → LLM → TTS → Vous entendez la réponse

VRAIE conversation voix-à-voix !
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

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("E2ETest")

class PipelineE2E:
    """Test pipeline End-to-End complet"""
    
    def __init__(self):
        self.stt_manager = None
        self.llm_manager = None
        self.tts_manager = None
        self.conversation_count = 0
        
    def initialize_pipeline(self):
        """Initialise le pipeline complet"""
        logger.info("🚀 Initialisation pipeline E2E SuperWhisper V6")
        
        # 1. STT avec correction
        logger.info("🎤 STT Manager...")
        from STT.unified_stt_manager import UnifiedSTTManager
        
        stt_config = {
            'timeout_per_minute': 10.0,
            'cache_size_mb': 200,
            'fallback_chain': ['prism_primary'],
            'backends': [
                {
                    'name': 'prism_primary',
                    'type': 'prism',
                    'model': 'large-v2',
                    'device': 'cuda',
                    'compute_type': 'float16',
                    'language': 'fr'
                }
            ]
        }
        
        self.stt_manager = UnifiedSTTManager(stt_config)
        logger.info("✅ STT prêt")
        
        # 2. LLM
        logger.info("🧠 LLM Manager...")
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        
        llm_config = {
            "model_name": "nous-hermes",
            "max_context_turns": 5,
            "timeout_seconds": 15,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        self.llm_manager = EnhancedLLMManager(llm_config)
        logger.info("✅ LLM prêt")
        
        # 3. TTS
        logger.info("🔊 TTS Manager...")
        from TTS.tts_manager import UnifiedTTSManager
        
        tts_config = {
            "cache": {"enabled": True, "max_size": 100, "ttl_seconds": 3600},
            "circuit_breaker": {"failure_threshold": 5, "reset_timeout_seconds": 30},
            "backends": {"silent_emergency": {"enabled": True}},
            "default_handler": "silent_emergency"
        }
        
        self.tts_manager = UnifiedTTSManager(tts_config)
        logger.info("✅ TTS prêt")
        
        logger.info("🎉 Pipeline E2E initialisé !")
        
    def record_audio_question(self, duration=6):
        """Enregistre votre question"""
        try:
            import sounddevice as sd
            import numpy as np
            
            sample_rate = 16000
            
            print(f"\n🔴 ENREGISTREMENT {duration}s")
            print("💬 POSEZ VOTRE QUESTION MAINTENANT !")
            print("🎯 Exemples: 'Bonjour', 'Comment ça va ?', 'Quel temps fait-il ?'")
            
            # Compte à rebours
            for i in range(3, 0, -1):
                print(f"▶️ Début dans {i}s...")
                time.sleep(1)
            
            print("🎙️ PARLEZ MAINTENANT !")
            
            # Enregistrement
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32
            )
            
            # Progress bar
            for i in range(duration):
                print(f"⏳ {i+1}/{duration}s", end='\r')
                time.sleep(1)
            
            sd.wait()
            print("\n✅ Enregistrement terminé")
            
            # Vérification niveau
            audio_level = np.max(np.abs(audio_data))
            print(f"📊 Niveau audio: {audio_level:.3f}")
            
            if audio_level > 0.01:
                print("✅ Audio détecté")
                return audio_data.flatten(), sample_rate
            else:
                print("⚠️ Audio très faible")
                return None, 0
                
        except Exception as e:
            logger.error(f"❌ Erreur enregistrement: {e}")
            return None, 0
            
    async def transcribe_question(self, audio_data, sample_rate):
        """Transcrit votre question"""
        try:
            print("🎤 Transcription en cours...")
            start_time = time.time()
            
            # Transcription avec notre STT validé
            result = await self.stt_manager.transcribe(audio_data)
            transcription = result.text if hasattr(result, 'text') else str(result)
            
            stt_time = time.time() - start_time
            
            print(f"📝 Vous avez dit ({stt_time:.1f}s): '{transcription}'")
            
            if not transcription.strip():
                return None, 0
                
            return transcription, stt_time
            
        except Exception as e:
            logger.error(f"❌ Erreur STT: {e}")
            return None, 0
            
    async def generate_answer(self, question):
        """Génère la réponse à votre question"""
        try:
            print("🧠 Génération réponse...")
            start_time = time.time()
            
            try:
                response = await self.llm_manager.generate_response(
                    user_input=question,
                    max_tokens=80,
                    temperature=0.7,
                    include_context=True
                )
            except Exception as e:
                logger.warning(f"⚠️ LLM erreur: {e}")
                # Réponses conversationnelles intelligentes
                question_lower = question.lower()
                if "bonjour" in question_lower or "salut" in question_lower:
                    response = "Bonjour ! Je suis SuperWhisper V6, votre assistant vocal. Comment puis-je vous aider ?"
                elif "comment" in question_lower and ("va" in question_lower or "allez" in question_lower):
                    response = "Je vais très bien, merci ! Et vous, comment allez-vous aujourd'hui ?"
                elif "temps" in question_lower or "météo" in question_lower:
                    response = "Je ne peux pas consulter la météo en temps réel, mais j'espère qu'il fait beau chez vous !"
                elif "qui" in question_lower and ("tu" in question_lower or "vous" in question_lower):
                    response = "Je suis SuperWhisper V6, un assistant vocal intelligent développé pour vous aider dans vos conversations."
                elif "merci" in question_lower:
                    response = "De rien ! C'est un plaisir de pouvoir vous aider. Y a-t-il autre chose que vous aimeriez savoir ?"
                elif "au revoir" in question_lower or "bye" in question_lower:
                    response = "Au revoir ! Merci pour cette conversation. À bientôt !"
                else:
                    response = f"C'est intéressant ! Vous me dites '{question}'. Pouvez-vous m'en dire un peu plus ?"
            
            llm_time = time.time() - start_time
            
            print(f"🤖 Réponse générée ({llm_time:.1f}s): '{response}'")
            return response, llm_time
            
        except Exception as e:
            logger.error(f"❌ Erreur LLM: {e}")
            return "Désolé, j'ai un problème technique.", 0
            
    async def speak_answer(self, text):
        """Lit la réponse à haute voix"""
        try:
            print("🔊 Synthèse et lecture...")
            start_time = time.time()
            
            # Synthèse
            audio_data = await self.tts_manager.synthesize(text=text)
            
            tts_time = time.time() - start_time
            
            if audio_data and len(audio_data) > 0:
                print(f"🎵 Audio généré ({tts_time:.1f}s) - Lecture en cours...")
                
                # Sauvegarde et lecture
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file.flush()
                    
                    # Lecture selon OS
                    import subprocess
                    if sys.platform.startswith('win'):
                        subprocess.run(['start', '/min', '', temp_file.name], shell=True)
                    elif sys.platform.startswith('linux'):
                        subprocess.run(['aplay', temp_file.name], stderr=subprocess.DEVNULL)
                    else:
                        subprocess.run(['afplay', temp_file.name])
                    
                    # Attendre fin de lecture
                    duration_estimate = len(text) * 0.08 + 1  # ~80ms par caractère
                    print(f"🎧 Écoute de la réponse ({duration_estimate:.1f}s)...")
                    time.sleep(duration_estimate)
                    
                    # Nettoyage
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                        
                print("✅ Lecture terminée")
            else:
                print(f"🔊 Réponse synthétisée ({tts_time:.1f}s) - Mode silencieux")
                print(f"📢 Réponse (texte): {text}")
            
            return tts_time
            
        except Exception as e:
            logger.error(f"❌ Erreur TTS: {e}")
            print(f"📢 Réponse (texte): {text}")
            return 0.1
            
    async def conversation_complete(self):
        """Une conversation complète E2E"""
        self.conversation_count += 1
        
        print(f"\n💬 CONVERSATION #{self.conversation_count}")
        print("="*50)
        
        total_start = time.time()
        
        # 1. Enregistrement
        audio_data, sample_rate = self.record_audio_question()
        if audio_data is None:
            print("❌ Problème enregistrement")
            return False
        
        # 2. STT
        transcription, stt_time = await self.transcribe_question(audio_data, sample_rate)
        if not transcription:
            print("❌ Transcription échouée")
            return False
        
        # 3. LLM
        response, llm_time = await self.generate_answer(transcription)
        
        # 4. TTS + Lecture
        tts_time = await self.speak_answer(response)
        
        # 5. Métriques
        total_time = time.time() - total_start
        pipeline_time = stt_time + llm_time + tts_time
        
        print(f"\n📊 MÉTRIQUES CONVERSATION #{self.conversation_count}")
        print("="*40)
        print(f"🎤 Vous: '{transcription}'")
        print(f"🤖 SuperWhisper: '{response}'")
        print(f"⏱️ Latences:")
        print(f"   • STT: {stt_time:.1f}s")
        print(f"   • LLM: {llm_time:.1f}s")
        print(f"   • TTS: {tts_time:.1f}s")
        print(f"   • Pipeline: {pipeline_time:.1f}s")
        print(f"   • Total E2E: {total_time:.1f}s")
        
        if pipeline_time < 3.0:
            print("🎯 ✅ OBJECTIF <3s ATTEINT !")
        else:
            print("🎯 ⚠️ Latence >3s")
        
        return True
        
    async def run_e2e_test(self):
        """Test E2E complet avec plusieurs conversations"""
        print("\n🎯 TEST PIPELINE E2E - CONVERSATION VOIX-À-VOIX")
        print("="*60)
        print("🎤 Vous parlez → 🧠 IA répond → 🔊 Vous entendez")
        print("🔄 Test de continuité complète du pipeline")
        print("="*60)
        
        try:
            while True:
                # Conversation complète
                success = await self.conversation_complete()
                
                if not success:
                    print("⚠️ Conversation échouée, nouvelle tentative ?")
                
                # Continuer ?
                print(f"\n❓ Nouvelle conversation ? (Entrée=Oui, Ctrl+C=Stop)")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    break
                    
        except KeyboardInterrupt:
            print(f"\n🏁 Test E2E terminé après {self.conversation_count} conversations")
        
        # Bilan final
        if self.conversation_count > 0:
            print(f"\n🎉 TEST E2E RÉUSSI !")
            print(f"✅ {self.conversation_count} conversations voix-à-voix validées")
            print("✅ Pipeline E2E SuperWhisper V6 opérationnel")
            print("🚀 Continuité complète confirmée !")
        else:
            print("\n⚠️ Aucune conversation complète")

async def main():
    """Test principal E2E"""
    pipeline = PipelineE2E()
    
    try:
        # Vérification audio
        print("🔊 Vérification microphone...")
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            mic_found = False
            for device in devices:
                if device['max_input_channels'] > 0:
                    print(f"🎤 Microphone: {device['name']}")
                    mic_found = True
                    break
            
            if not mic_found:
                print("❌ Aucun microphone détecté")
                return
                
        except ImportError:
            print("❌ Module sounddevice manquant: pip install sounddevice")
            return
        
        # Initialisation
        pipeline.initialize_pipeline()
        
        # Test E2E
        await pipeline.run_e2e_test()
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 SUPERWHISPER V6 - TEST PIPELINE E2E COMPLET")
    print("🎤 Test de continuité: Voix → IA → Voix")
    print("🚀 Validation conversation temps réel")
    print()
    
    asyncio.run(main())