#!/usr/bin/env python3
"""
🎉 DÉMONSTRATION SUPERWHISPER V6 COMPLET
Pipeline voix-à-voix avec streaming microphone temps réel

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

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
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Imports après configuration GPU
try:
    import torch
    from STT.unified_stt_manager import UnifiedSTTManager
    from STT.streaming_microphone_manager import StreamingMicrophoneManager, SpeechSegment
    from TTS.tts_manager import TTSManager
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    sys.exit(1)

# =============================================================================
# VALIDATION RTX 3090 OBLIGATOIRE
# =============================================================================
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

# =============================================================================
# PIPELINE SUPERWHISPER V6 COMPLET
# =============================================================================
class SuperWhisperV6Pipeline:
    """Pipeline complet SuperWhisper V6 : STT → LLM → TTS"""
    
    def __init__(self):
        validate_rtx3090_configuration()
        
        self.stt_manager = None
        self.tts_manager = None
        self.mic_manager = None
        
        # Statistiques pipeline
        self.stats = {
            'conversations': 0,
            'total_stt_time': 0.0,
            'total_llm_time': 0.0,
            'total_tts_time': 0.0,
            'total_pipeline_time': 0.0
        }
    
    async def initialize(self):
        """Initialisation complète du pipeline"""
        print("\n🚀 INITIALISATION SUPERWHISPER V6 PIPELINE")
        print("="*60)
        
        # Initialisation STT
        print("\n🎤 Initialisation STT Manager...")
        stt_config = {
            'timeout_per_minute': 10.0,
            'max_retries': 3,
            'cache_enabled': True,
            'circuit_breaker_enabled': True,
            'fallback_chain': ['prism_primary'],
            'backends': [
                {
                    'name': 'prism_primary',
                    'type': 'prism',
                    'model': 'large-v2',
                    'compute_type': 'float16',
                    'language': 'fr',
                    'beam_size': 5,
                    'vad_filter': True
                }
            ]
        }
        
        self.stt_manager = UnifiedSTTManager(config=stt_config)
        print("✅ STT Manager initialisé")
        
        # Initialisation TTS
        print("\n🔊 Initialisation TTS Manager...")
        tts_config_path = Path("config/tts.yaml")
        if tts_config_path.exists():
            self.tts_manager = TTSManager(config_path=str(tts_config_path))
        else:
            # Configuration par défaut
            self.tts_manager = TTSManager()
        print("✅ TTS Manager initialisé")
        
        # Initialisation Streaming Microphone
        print("\n🎙️ Initialisation Streaming Microphone...")
        self.mic_manager = StreamingMicrophoneManager(
            stt_manager=self.stt_manager,
            on_transcription=self.on_speech_transcribed
        )
        print("✅ Streaming Microphone initialisé")
        
        print("\n🎉 SUPERWHISPER V6 PIPELINE PRÊT")
        print("="*60)
    
    def on_speech_transcribed(self, text: str, segment: SpeechSegment):
        """Callback appelé quand la parole est transcrite"""
        if not text.strip():
            return
        
        print(f"\n🗣️ UTILISATEUR: '{text}'")
        
        # Démarrer traitement pipeline asynchrone
        asyncio.create_task(self.process_conversation(text, segment))
    
    async def process_conversation(self, user_text: str, segment: SpeechSegment):
        """Traitement complet conversation : STT → LLM → TTS"""
        conversation_start = time.time()
        
        try:
            # Étape 1: STT déjà fait
            stt_time = (time.time() - segment.end_ts)
            print(f"   ✅ STT: {stt_time*1000:.0f}ms")
            
            # Étape 2: LLM (simulation simple pour démo)
            llm_start = time.time()
            await asyncio.sleep(0.1)  # Simulation traitement LLM
            
            # Réponse simple basée sur le contenu
            if "bonjour" in user_text.lower():
                ai_response = "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
            elif "comment" in user_text.lower() and "allez" in user_text.lower():
                ai_response = "Je vais très bien, merci ! Et vous ?"
            elif "merci" in user_text.lower():
                ai_response = "Je vous en prie ! C'est un plaisir de vous aider."
            elif "au revoir" in user_text.lower():
                ai_response = "Au revoir ! À bientôt !"
            else:
                ai_response = f"J'ai bien entendu : {user_text}. C'est très intéressant !"
            
            llm_time = time.time() - llm_start
            print(f"   🤖 LLM: {llm_time*1000:.0f}ms - '{ai_response}'")
            
            # Étape 3: TTS
            tts_start = time.time()
            audio_path = await self.tts_manager.synthesize_async(
                text=ai_response,
                voice="fr-FR-DeniseNeural",
                output_path=f"temp_response_{int(time.time())}.wav"
            )
            tts_time = time.time() - tts_start
            print(f"   🔊 TTS: {tts_time*1000:.0f}ms - Audio généré")
            
            # Lecture audio (optionnel)
            if audio_path and Path(audio_path).exists():
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    
                    # Attendre fin lecture
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)
                    
                    # Nettoyage
                    Path(audio_path).unlink(missing_ok=True)
                    print(f"   🎵 Audio joué et nettoyé")
                    
                except ImportError:
                    print(f"   ⚠️ pygame non disponible - audio sauvé: {audio_path}")
                except Exception as e:
                    print(f"   ⚠️ Erreur lecture audio: {e}")
            
            # Statistiques
            total_time = time.time() - conversation_start
            self.stats['conversations'] += 1
            self.stats['total_stt_time'] += stt_time
            self.stats['total_llm_time'] += llm_time
            self.stats['total_tts_time'] += tts_time
            self.stats['total_pipeline_time'] += total_time
            
            print(f"   ⏱️ TOTAL PIPELINE: {total_time*1000:.0f}ms")
            print(f"   📊 Conversation #{self.stats['conversations']}")
            
        except Exception as e:
            print(f"   ❌ Erreur pipeline: {e}")
    
    async def run_demo(self):
        """Démonstration interactive SuperWhisper V6"""
        print("\n🎯 DÉMONSTRATION SUPERWHISPER V6 INTERACTIVE")
        print("="*60)
        print("📋 Instructions:")
        print("   1. Parlez clairement au microphone")
        print("   2. SuperWhisper V6 va :")
        print("      - Transcrire votre parole (STT)")
        print("      - Générer une réponse (LLM)")
        print("      - Synthétiser la réponse (TTS)")
        print("      - Jouer l'audio de réponse")
        print("   3. Testez différentes phrases :")
        print("      - 'Bonjour'")
        print("      - 'Comment allez-vous ?'")
        print("      - 'Merci beaucoup'")
        print("      - 'Au revoir'")
        print("   4. Appuyez Ctrl+C pour arrêter")
        print("\n⏰ Démarrage dans 3 secondes...")
        
        await asyncio.sleep(3)
        
        print("\n🎙️ SUPERWHISPER V6 DÉMARRÉ - PIPELINE VOIX-À-VOIX ACTIF")
        print("🛑 Appuyez Ctrl+C pour arrêter")
        print("-" * 60)
        
        try:
            await self.mic_manager.run()
        except KeyboardInterrupt:
            print("\n🛑 Démonstration arrêtée par utilisateur")
        finally:
            self.print_final_stats()
    
    def print_final_stats(self):
        """Affichage statistiques finales"""
        if self.stats['conversations'] == 0:
            print("\n📊 Aucune conversation traitée")
            return
        
        avg_stt = (self.stats['total_stt_time'] / self.stats['conversations']) * 1000
        avg_llm = (self.stats['total_llm_time'] / self.stats['conversations']) * 1000
        avg_tts = (self.stats['total_tts_time'] / self.stats['conversations']) * 1000
        avg_total = (self.stats['total_pipeline_time'] / self.stats['conversations']) * 1000
        
        print(f"\n📊 STATISTIQUES FINALES SUPERWHISPER V6")
        print("="*50)
        print(f"🗣️ Conversations traitées: {self.stats['conversations']}")
        print(f"🎤 STT moyen: {avg_stt:.0f}ms")
        print(f"🤖 LLM moyen: {avg_llm:.0f}ms")
        print(f"🔊 TTS moyen: {avg_tts:.0f}ms")
        print(f"⏱️ Pipeline total moyen: {avg_total:.0f}ms")
        print(f"🎯 Performance: Pipeline voix-à-voix < {avg_total/1000:.1f}s")

# =============================================================================
# MAIN
# =============================================================================
async def main():
    """Point d'entrée principal"""
    
    print("🎉 SUPERWHISPER V6 - DÉMONSTRATION PIPELINE COMPLET")
    print("="*60)
    print("🎯 Pipeline: Microphone → STT → LLM → TTS → Audio")
    print("🎮 GPU: RTX 3090 exclusif")
    print("🎙️ Microphone: RODE NT-USB détection automatique")
    print("🔊 Audio: Synthèse et lecture automatique")
    print()
    
    try:
        # Initialisation pipeline
        pipeline = SuperWhisperV6Pipeline()
        await pipeline.initialize()
        
        # Démonstration interactive
        await pipeline.run_demo()
        
    except Exception as e:
        print(f"❌ Erreur démonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎉 SuperWhisper V6 - Pipeline Voix-à-Voix Complet")
    print("🚀 Streaming microphone + STT + LLM + TTS intégré")
    print()
    
    asyncio.run(main()) 