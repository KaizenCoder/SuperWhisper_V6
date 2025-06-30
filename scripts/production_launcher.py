#!/usr/bin/env python3
"""
SuperWhisper V6 - Production Launcher
Lanceur sécurisé pour environnement de production
"""

import os
import sys
import pathlib
import asyncio
import logging
import signal
import json
from datetime import datetime
from typing import Dict, Any

def _setup_portable_environment():
    current_file = pathlib.Path(__file__).resolve()
    
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    os.chdir(project_root)
    
    # Configuration GPU RTX 3090
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    return project_root

_PROJECT_ROOT = _setup_portable_environment()

# Configuration logging production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(f'production_logs_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProductionLauncher")

class ProductionPipeline:
    """Pipeline SuperWhisper V6 pour environnement de production"""
    
    def __init__(self):
        self.running = False
        self.components = {}
        self.session_stats = {
            'start_time': None,
            'conversations': 0,
            'stt_calls': 0,
            'llm_calls': 0,
            'tts_calls': 0,
            'errors': 0
        }
        
        # Gestion signaux système
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Gestionnaire arrêt propre"""
        logger.info(f"🛑 Signal {signum} reçu - Arrêt gracieux...")
        self.running = False

    async def initialize_components(self):
        """Initialisation de tous les composants"""
        logger.info("🚀 Initialisation composants production...")
        
        try:
            # STT Manager
            from STT.unified_stt_manager import UnifiedSTTManager
            from STT.streaming_microphone_manager import StreamingMicrophoneManager
            
            stt_config = {
                'timeout_per_minute': 10.0,
                'cache_size_mb': 200,
                'cache_ttl': 7200,
                'max_retries': 3,
                'fallback_chain': ['prism_primary']
            }
            
            self.components['stt'] = UnifiedSTTManager(stt_config)
            self.components['microphone'] = StreamingMicrophoneManager()
            logger.info("✅ STT Manager initialisé")
            
            # LLM Manager
            from LLM.llm_manager_enhanced import EnhancedLLMManager
            
            llm_config = {
                'model': 'nous-hermes-2-mistral-7b-dpo:latest',
                'use_ollama': True,
                'timeout': 30.0,
                'max_tokens': 150,
                'temperature': 0.7
            }
            
            self.components['llm'] = EnhancedLLMManager(llm_config)
            await self.components['llm'].initialize()
            logger.info("✅ LLM Manager initialisé")
            
            # TTS Manager
            from TTS.tts_manager import UnifiedTTSManager
            import yaml
            
            with open('config/tts.yaml', 'r') as f:
                tts_config = yaml.safe_load(f)
            
            self.components['tts'] = UnifiedTTSManager(tts_config)
            logger.info("✅ TTS Manager initialisé")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation: {e}")
            return False

    async def pre_flight_check(self):
        """Vérifications pré-vol avant démarrage"""
        logger.info("🔍 Vérifications pré-vol...")
        
        try:
            # Test STT
            import numpy as np
            sample_rate = 16000
            duration = 1.0
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            stt_result = await self.components['stt'].transcribe_pcm(audio_bytes, sample_rate)
            logger.info("✅ Test STT réussi")
            
            # Test LLM
            llm_response = await self.components['llm'].generate_response(
                "Test système", max_tokens=20
            )
            logger.info("✅ Test LLM réussi")
            
            # Test TTS
            tts_result = await self.components['tts'].synthesize(text="Test système")
            if tts_result.success:
                logger.info("✅ Test TTS réussi")
            else:
                logger.error(f"❌ Test TTS échoué: {tts_result.error}")
                return False
            
            # Test microphone
            devices = self.components['microphone'].get_audio_devices()
            rode_count = len([d for d in devices if 'rode' in d['name'].lower() or 'nt-usb' in d['name'].lower()])
            
            if rode_count > 0:
                logger.info(f"✅ {rode_count} RODE NT-USB détectés")
            else:
                logger.warning("⚠️ Aucun RODE NT-USB - Utilisation microphone par défaut")
            
            logger.info("🎯 Tous les tests pré-vol réussis")
            return True
            
        except Exception as e:
            logger.error(f"❌ Échec vérifications pré-vol: {e}")
            return False

    async def conversation_cycle(self):
        """Cycle principal de conversation"""
        logger.info("🎤 Démarrage cycle conversation...")
        
        try:
            # Initialisation microphone
            await self.components['microphone'].initialize()
            logger.info("🎧 Microphone initialisé - En attente de voix...")
            
            while self.running:
                try:
                    # 1. Capture audio
                    logger.info("👂 Écoute en cours...")
                    audio_data = await self.components['microphone'].capture_voice_segment()
                    
                    if not audio_data:
                        await asyncio.sleep(0.1)
                        continue
                    
                    self.session_stats['conversations'] += 1
                    conversation_start = asyncio.get_event_loop().time()
                    
                    logger.info(f"🎵 Audio capturé: {len(audio_data)} bytes")
                    
                    # 2. Transcription STT
                    logger.info("🎤 Transcription en cours...")
                    stt_start = asyncio.get_event_loop().time()
                    
                    transcription = await self.components['stt'].transcribe_pcm(
                        audio_data, sample_rate=16000
                    )
                    
                    stt_time = (asyncio.get_event_loop().time() - stt_start) * 1000
                    self.session_stats['stt_calls'] += 1
                    
                    if not transcription or transcription.strip() == "":
                        logger.info("🔇 Transcription vide - Ignoré")
                        continue
                    
                    logger.info(f"📝 Transcription ({stt_time:.1f}ms): {transcription}")
                    
                    # 3. Génération LLM
                    logger.info("🧠 Génération réponse...")
                    llm_start = asyncio.get_event_loop().time()
                    
                    response = await self.components['llm'].generate_response(
                        transcription, max_tokens=150
                    )
                    
                    llm_time = (asyncio.get_event_loop().time() - llm_start) * 1000
                    self.session_stats['llm_calls'] += 1
                    
                    logger.info(f"💭 Réponse LLM ({llm_time:.1f}ms): {response[:50]}...")
                    
                    # 4. Synthèse TTS
                    logger.info("🔊 Synthèse vocale...")
                    tts_start = asyncio.get_event_loop().time()
                    
                    tts_result = await self.components['tts'].synthesize(text=response)
                    
                    tts_time = (asyncio.get_event_loop().time() - tts_start) * 1000
                    self.session_stats['tts_calls'] += 1
                    
                    if not tts_result.success:
                        logger.error(f"❌ Erreur TTS: {tts_result.error}")
                        self.session_stats['errors'] += 1
                        continue
                    
                    logger.info(f"🎵 Audio TTS généré ({tts_time:.1f}ms): {len(tts_result.audio_data)} bytes")
                    
                    # 5. Anti-feedback obligatoire
                    logger.info("⏸️ Pause anti-feedback 3s...")
                    await asyncio.sleep(3)
                    
                    # 6. Statistiques conversation
                    total_time = (asyncio.get_event_loop().time() - conversation_start) * 1000
                    
                    logger.info("=" * 60)
                    logger.info(f"📊 CONVERSATION #{self.session_stats['conversations']} TERMINÉE")
                    logger.info(f"   STT: {stt_time:.1f}ms | LLM: {llm_time:.1f}ms | TTS: {tts_time:.1f}ms")
                    logger.info(f"   Total: {total_time:.1f}ms")
                    logger.info("=" * 60)
                    
                except Exception as e:
                    logger.error(f"❌ Erreur cycle conversation: {e}")
                    self.session_stats['errors'] += 1
                    await asyncio.sleep(1)  # Pause avant retry
                    
        except Exception as e:
            logger.error(f"❌ Erreur fatale conversation: {e}")
            self.running = False

    async def cleanup(self):
        """Nettoyage ressources"""
        logger.info("🧹 Nettoyage en cours...")
        
        try:
            if 'microphone' in self.components:
                await self.components['microphone'].cleanup()
            
            if 'llm' in self.components:
                await self.components['llm'].cleanup()
            
            # Nettoyage fichiers audio temporaires
            import glob
            temp_files = glob.glob("tts_output_*.wav")
            for file in temp_files:
                try:
                    os.remove(file)
                except:
                    pass
            
            logger.info("✅ Nettoyage terminé")
            
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage: {e}")

    def print_session_stats(self):
        """Affichage statistiques session"""
        duration = (datetime.now() - self.session_stats['start_time']).total_seconds()
        
        print("\n" + "="*60)
        print("📊 STATISTIQUES SESSION SuperWhisper V6")
        print("="*60)
        print(f"⏱️ Durée session: {duration:.1f}s")
        print(f"💬 Conversations: {self.session_stats['conversations']}")
        print(f"🎤 Appels STT: {self.session_stats['stt_calls']}")
        print(f"🧠 Appels LLM: {self.session_stats['llm_calls']}")
        print(f"🔊 Appels TTS: {self.session_stats['tts_calls']}")
        print(f"❌ Erreurs: {self.session_stats['errors']}")
        
        if self.session_stats['conversations'] > 0:
            avg_per_min = (self.session_stats['conversations'] / duration) * 60
            print(f"📈 Moyenne: {avg_per_min:.1f} conversations/min")
        
        print("="*60)

    async def run(self):
        """Méthode principale d'exécution"""
        print("🚀 SuperWhisper V6 - Mode Production")
        print("="*60)
        
        self.session_stats['start_time'] = datetime.now()
        
        try:
            # Initialisation
            if not await self.initialize_components():
                logger.error("❌ Échec initialisation - Arrêt")
                return False
            
            # Vérifications pré-vol
            if not await self.pre_flight_check():
                logger.error("❌ Échec vérifications pré-vol - Arrêt")
                return False
            
            # Démarrage pipeline
            logger.info("🎯 SuperWhisper V6 OPÉRATIONNEL en production")
            logger.info("   Parlez dans le microphone pour démarrer une conversation")
            logger.info("   Ctrl+C pour arrêter proprement")
            
            self.running = True
            await self.conversation_cycle()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé par utilisateur")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur fatale: {e}")
            return False
        finally:
            await self.cleanup()
            self.print_session_stats()

async def main():
    """Point d'entrée principal"""
    pipeline = ProductionPipeline()
    
    try:
        success = await pipeline.run()
        
        if success:
            print("\n✅ SuperWhisper V6 arrêté proprement")
            sys.exit(0)
        else:
            print("\n❌ SuperWhisper V6 arrêté avec erreur")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())