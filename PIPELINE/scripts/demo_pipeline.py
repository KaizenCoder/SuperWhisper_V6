#!/usr/bin/env python3
"""
Démonstration Pipeline SuperWhisper V6 - Task 18.8
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Script démonstration utilisant le code OBLIGATOIRE du prompt

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
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 Demo Pipeline: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter le répertoire parent au path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# IMPORTS ET CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s – %(levelname)s – %(name)s – %(message)s",
)
LOGGER = logging.getLogger("DemoPipeline")

# =============================================================================
# FONCTION BOOTSTRAP OBLIGATOIRE DU PROMPT
# =============================================================================

async def _bootstrap(cfg_path: Optional[str] = None):
    """Bootstrap function from prompt - MANDATORY CODE"""
    import yaml
    cfg: Dict[str, Any] = {}
    if cfg_path and Path(cfg_path).exists():
        cfg = yaml.safe_load(Path(cfg_path).read_text())

    # Import components
    from STT.unified_stt_manager_optimized import OptimizedUnifiedSTTManager
    from TTS.tts_manager import UnifiedTTSManager
    from PIPELINE.pipeline_orchestrator import PipelineOrchestrator

    # ✅ CORRECTION: Use OptimizedUnifiedSTTManager
    stt = OptimizedUnifiedSTTManager(cfg.get("stt", {}))
    tts = UnifiedTTSManager(cfg.get("tts", {}))
    orchestrator = PipelineOrchestrator(
        stt,
        tts,
        llm_endpoint=cfg.get("pipeline", {}).get("llm_endpoint", "http://localhost:8000"),
        metrics_enabled=cfg.get("pipeline", {}).get("enable_metrics", False),
    )
    await orchestrator.start()

# =============================================================================
# DÉMONSTRATION INTERACTIVE
# =============================================================================

class PipelineDemo:
    """Démonstration interactive utilisant le code obligatoire du prompt"""
    
    def __init__(self):
        self.config_path = "PIPELINE/config/pipeline.yaml"
        
    async def run_demo(self):
        """Exécuter démonstration complète"""
        print("\n" + "="*60)
        print("🚀 DÉMONSTRATION SUPERWHISPER V6 PIPELINE")
        print("🚨 CODE OBLIGATOIRE DU PROMPT UTILISÉ")
        print("="*60)
        
        try:
            await self._show_menu()
        except KeyboardInterrupt:
            print("\n🛑 Démonstration interrompue par l'utilisateur")
        except Exception as e:
            LOGGER.error(f"❌ Erreur démonstration: {e}")
            print(f"❌ Erreur: {e}")
    
    async def _show_menu(self):
        """Menu principal démonstration"""
        while True:
            print("\n🎯 OPTIONS DÉMONSTRATION")
            print("-" * 40)
            print("1. 🚀 Démarrer pipeline complet (code obligatoire)")
            print("2. 🧪 Test validation environnement")
            print("3. 📊 Afficher configuration")
            print("4. 👋 Quitter")
            
            try:
                choice = input("\n👉 Votre choix (1-4): ").strip()
                
                if choice == "1":
                    await self._start_pipeline_obligatoire()
                elif choice == "2":
                    await self._test_environment()
                elif choice == "3":
                    self._show_config()
                elif choice == "4":
                    print("👋 Au revoir!")
                    break
                else:
                    print("❌ Choix invalide. Utilisez 1-4.")
                    
            except KeyboardInterrupt:
                print("\n🛑 Démonstration interrompue")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
    
    async def _start_pipeline_obligatoire(self):
        """Démarrer pipeline avec code obligatoire du prompt"""
        print("\n🚀 DÉMARRAGE PIPELINE - CODE OBLIGATOIRE")
        print("="*50)
        print("⚠️ Assurez-vous que:")
        print("  - Votre microphone est connecté")
        print("  - Le serveur LLM est démarré (http://localhost:8000)")
        print("  - Appuyez sur Ctrl+C pour arrêter")
        print("\n🎤 Parlez dans votre microphone...")
        
        try:
            # Utiliser la fonction bootstrap OBLIGATOIRE du prompt
            await _bootstrap(self.config_path)
        except KeyboardInterrupt:
            print("\n🛑 Pipeline arrêté")
        except Exception as e:
            print(f"❌ Erreur pipeline: {e}")
            LOGGER.error(f"Pipeline error: {e}")
    
    async def _test_environment(self):
        """Test validation environnement"""
        print("\n🧪 TEST VALIDATION ENVIRONNEMENT")
        print("-" * 40)
        
        # Test GPU
        print("🔍 Validation GPU RTX 3090...")
        try:
            from PIPELINE.scripts.assert_gpu_env import main as validate_gpu
            validate_gpu()
            print("✅ GPU RTX 3090 validée")
        except Exception as e:
            print(f"❌ Erreur validation GPU: {e}")
        
        # Test audio
        print("🔍 Validation devices audio...")
        try:
            from PIPELINE.scripts.validate_audio_devices import main as validate_audio
            validate_audio()
            print("✅ Devices audio validés")
        except Exception as e:
            print(f"⚠️ Avertissement audio: {e}")
        
        # Test LLM
        print("🔍 Test serveur LLM...")
        try:
            from PIPELINE.scripts.start_llm import main as validate_llm
            await validate_llm()
            print("✅ Serveur LLM accessible")
        except Exception as e:
            print(f"⚠️ Avertissement LLM: {e}")
            print("   Le pipeline utilisera les fallbacks")
        
        print("\n✅ Tests environnement terminés")
    
    def _show_config(self):
        """Afficher configuration actuelle"""
        print("\n📊 CONFIGURATION PIPELINE")
        print("-" * 30)
        
        try:
            import yaml
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                print(f"📁 Fichier: {self.config_path}")
                print(f"🎤 STT Backend: {config.get('stt', {}).get('primary_backend', 'N/A')}")
                print(f"🔊 TTS Backend: {config.get('tts', {}).get('primary_backend', 'N/A')}")
                print(f"🤖 LLM Endpoint: {config.get('pipeline', {}).get('llm_endpoint', 'N/A')}")
                print(f"📊 Métriques: {config.get('pipeline', {}).get('enable_metrics', False)}")
                print(f"🎮 GPU: RTX 3090 (CUDA:1) - OBLIGATOIRE")
            else:
                print(f"⚠️ Configuration non trouvée: {self.config_path}")
                print("📋 Configuration par défaut sera utilisée")
        except Exception as e:
            print(f"❌ Erreur lecture configuration: {e}")

# =============================================================================
# SCRIPT ENTRY POINT - CODE OBLIGATOIRE DU PROMPT
# =============================================================================

async def main():
    """Point d'entrée principal - utilise code obligatoire"""
    demo = PipelineDemo()
    await demo.run_demo()

def signal_handler(signum, frame):
    """Gestionnaire signal pour arrêt propre"""
    print("\n🛑 Signal d'arrêt reçu...")
    sys.exit(0)

if __name__ == "__main__":
    # Gestionnaire signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Optimisation uvloop comme dans le prompt obligatoire
        try:
            import uvloop
            uvloop.install()
            LOGGER.info("✅ uvloop enabled for enhanced performance")
        except ImportError:
            LOGGER.info("uvloop not available – fallback to asyncio event‑loop")

        # Démarrer démonstration
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("👋 Keyboard interrupt – exit")
    except Exception as e:
        LOGGER.error("❌ Demo startup error: %s", e)
        sys.exit(1) 