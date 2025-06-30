#!/usr/bin/env python3
"""
Luxa - SuperWhisper_V6 Assistant v1.1
======================================

Assistant vocal intelligent avec pipeline STT → LLM → TTS

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

import argparse
import asyncio
import os
import sys
import time
import logging
from pathlib import Path
import yaml

# Configuration du logger
logger = logging.getLogger(__name__)
# Imports à ajouter/modifier
from STT.stt_manager_robust import RobustSTTManager
from STT.vad_manager import OptimizedVADManager
from LLM.llm_manager_enhanced import EnhancedLLMManager
from TTS.tts_handler import TTSHandler

# Ajouter le répertoire courant au PYTHONPATH pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Orchestrator.master_handler_robust import RobustMasterHandler
import numpy as np

def parse_arguments():
    """Parse les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Luxa - Assistant Vocal Intelligent v1.1"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["cli", "web", "api"],
        default="cli",
        help="Mode d'interface (défaut: cli)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port pour modes web/api (défaut: 8080)"
    )
    
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Fichier de configuration (défaut: config/settings.yaml)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Active le mode debug"
    )
    
    return parser.parse_args()

# Dans la fonction main() ou setup_components()
async def setup_stt_components(config):
    """Configuration des composants STT avec le nouveau manager"""
    
    # Initialisation VAD si configuré
    vad_manager = None
    if config.get('vad', {}).get('enabled', True):
        try:
            vad_manager = OptimizedVADManager()
            await vad_manager.initialize()
            logger.info("VAD Manager initialisé")
        except Exception as e:
            logger.warning(f"VAD non disponible: {e}")
    
    # Initialisation STT Manager Robuste
    stt_manager = RobustSTTManager(config['stt'], vad_manager=vad_manager)
    await stt_manager.initialize()
    
    logger.info(f"STT Manager initialisé sur {stt_manager.device}")
    return stt_manager

async def setup_llm_components(config):
    """Configuration des composants LLM avec le nouveau manager"""
    
    # Initialisation LLM Manager Enhanced
    llm_manager = EnhancedLLMManager(config['llm'])
    await llm_manager.initialize()
    
    logger.info("LLM Manager Enhanced initialisé avec succès")
    return llm_manager

async def run_cli_mode(handler):
    """Mode CLI interactif"""
    print("\n🎤 Mode CLI - Assistant Vocal")
    print("Commands: 'quit' pour quitter, 'status' pour le statut")
    print("=" * 50)
    
    try:
        while True:
            try:
                user_input = input("\n🗣️ Parlez (ou tapez): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Au revoir!")
                    break
                    
                elif user_input.lower() == 'status':
                    health = handler.get_health_status()
                    print(f"\n📊 Statut: {health['status']}")
                    print(f"Requêtes traitées: {health['performance']['requests_processed']}")
                    print(f"Latence moyenne: {health['performance']['avg_latency_ms']:.1f}ms")
                    continue
                    
                elif user_input.lower() == 'test':
                    # Test avec audio synthétique
                    print("🧪 Test avec audio synthétique...")
                    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
                    result = await handler.process_audio_safe(test_audio)
                    
                    print(f"✅ Résultat: {result['text']}")
                    print(f"⏱️ Latence: {result['latency_ms']:.1f}ms")
                    print(f"🎯 Succès: {result['success']}")
                    continue
                    
                if not user_input:
                    continue
                    
                print("📝 Traitement en cours...")
                
                # Pour l'instant, simuler avec du texte
                # Dans une vraie implémentation, on capturerait l'audio
                result = {
                    "success": True,
                    "text": f"Vous avez dit: {user_input}",
                    "latency_ms": 50
                }
                
                print(f"🎯 Réponse: {result['text']}")
                
            except KeyboardInterrupt:
                print("\n👋 Arrêt demandé...")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                
    except Exception as e:
        print(f"❌ Erreur CLI: {e}")

async def run_web_mode(handler, port):
    """Mode web (placeholder)"""
    print(f"🌐 Mode Web sur port {port}")
    print("⚠️ Interface web non implémentée dans cette version")
    
    # Placeholder pour serveur web
    print("Appuyez sur Ctrl+C pour arrêter...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Serveur web arrêté")

async def run_api_mode(handler, port):
    """Mode API REST (placeholder)"""
    print(f"🔌 Mode API REST sur port {port}")
    print("⚠️ API REST non implémentée dans cette version")
    
    # Placeholder pour API REST
    print("Appuyez sur Ctrl+C pour arrêter...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("🛑 API REST arrêtée")

def print_banner():
    """Affiche la bannière Luxa v1.1"""
    banner = """
    ██╗     ██╗   ██╗██╗  ██╗ █████╗ 
    ██║     ██║   ██║╚██╗██╔╝██╔══██╗
    ██║     ██║   ██║ ╚███╔╝ ███████║
    ██║     ██║   ██║ ██╔██╗ ██╔══██║
    ███████╗╚██████╔╝██╔╝ ██╗██║  ██║
    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
    
    🎤 Assistant Vocal Intelligent v1.1
    SuperWhisper_V6 - STT | LLM | TTS
    """
    print(banner)

async def main():
    """Fonction principale pour exécuter la boucle de l'assistant."""
    print("🚀 Démarrage de l'assistant vocal LUXA (MVP P0)...")

    # 1. Charger la configuration
    try:
        with open("Config/mvp_settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ ERREUR: Le fichier 'Config/mvp_settings.yaml' est introuvable.")
        return

    # 2. Initialiser les modules
    try:
        print("🔧 Initialisation des modules...")
        
        # Initialisation composants avec nouveaux managers
        stt_handler = await setup_stt_components(config)
        llm_handler = await setup_llm_components(config)
        tts_handler = TTSHandler(config['tts'])
        print("✅ Tous les modules sont initialisés!")
    except Exception as e:
        print(f"❌ ERREUR lors de l'initialisation: {e}")
        print(f"   Détails: {str(e)}")
        return

    # 3. Boucle principale de l'assistant
    print("\n🎯 Assistant vocal LUXA prêt!")
    print("Appuyez sur Ctrl+C pour arrêter")
    
    try:
        while True:
            print("\n" + "="*50)
            input("Appuyez sur Entrée pour commencer l'écoute...")
            
            # Pipeline STT → LLM → TTS
            try:
                total_start_time = time.perf_counter()
                
                # Étape STT
                stt_start_time = time.perf_counter()
                transcription = stt_handler.listen_and_transcribe(duration=7)
                stt_latency = time.perf_counter() - stt_start_time

                if transcription and transcription.strip():
                    print(f"📝 Transcription: '{transcription}'")
                    
                    # Étape LLM
                    llm_start_time = time.perf_counter()
                    response = await llm_handler.generate_response(transcription)
                    llm_latency = time.perf_counter() - llm_start_time

                    # Étape TTS
                    tts_start_time = time.perf_counter()
                    if response and response.strip():
                        tts_handler.speak(response)
                    tts_latency = time.perf_counter() - tts_start_time
                    
                    total_latency = time.perf_counter() - total_start_time
                    
                    print("\n--- 📊 Rapport de Latence ---")
                    print(f"  - STT: {stt_latency:.3f}s")
                    print(f"  - LLM: {llm_latency:.3f}s")
                    print(f"  - TTS: {tts_latency:.3f}s")
                    print(f"  - TOTAL: {total_latency:.3f}s")
                    print("----------------------------\n")

                else:
                    print("Aucun texte intelligible n'a été transcrit, nouvelle écoute...")
                    
            except Exception as e:
                print(f"❌ Erreur dans le pipeline: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n🛑 Arrêt de l'assistant vocal LUXA")
    finally:
        # Nettoyage des ressources
        try:
            if 'stt_handler' in locals():
                await stt_handler.cleanup()
            if 'llm_handler' in locals():
                await llm_handler.cleanup()
            print("✅ Nettoyage terminé")
        except Exception as e:
            print(f"⚠️ Erreur lors du nettoyage: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 