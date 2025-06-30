#!/usr/bin/env python3
"""
🎤 TEST PIPELINE RÉEL SIMPLE - SUPERWHISPER V6
==============================================
Test validation humaine RÉELLE avec PipelineOrchestrator existant

MISSION:
- Utiliser PipelineOrchestrator déjà validé
- Test microphone → réponse vocale
- Validation humaine expérience utilisateur

Usage: python scripts/test_pipeline_reel_simple.py

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
import logging
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajout du chemin pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "PIPELINE"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipeline_reel_simple")

async def test_pipeline_reel():
    """Test pipeline réel avec PipelineOrchestrator"""
    try:
        # Import de la fonction bootstrap validée
        from pipeline_orchestrator import _bootstrap
        
        logger.info("🚀 DÉMARRAGE TEST PIPELINE RÉEL SIMPLE")
        logger.info("="*60)
        
        # Utiliser la fonction bootstrap qui gère tout automatiquement
        logger.info("🔧 Bootstrap pipeline avec configuration validée...")
        pipeline = await _bootstrap(cfg_path="PIPELINE/config/pipeline.yaml")
        
        if pipeline is None:
            logger.error("❌ Échec bootstrap pipeline")
            return False
        
        logger.info("✅ Pipeline bootstrappé avec succès !")
        
        logger.info("✅ Pipeline démarré avec succès !")
        logger.info("\n🎤 MODE TEST INTERACTIF")
        logger.info("Le pipeline est maintenant actif.")
        logger.info("Parlez dans votre microphone RODE NT-USB...")
        logger.info("Appuyez sur Ctrl+C pour arrêter")
        
        # Laisser le pipeline tourner
        try:
            while True:
                await asyncio.sleep(1)
                
                # Afficher métriques périodiquement
                if hasattr(pipeline, 'metrics'):
                    metrics = pipeline.metrics
                    logger.info(f"📊 Conversations: {len(pipeline.conversation_history)}")
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Arrêt demandé par utilisateur")
        
        # Arrêter pipeline
        logger.info("🔄 Arrêt pipeline...")
        await pipeline.stop()
        
        logger.info("✅ Pipeline arrêté proprement")
        
        # Afficher résultats
        if hasattr(pipeline, 'conversation_history') and pipeline.conversation_history:
            logger.info(f"\n📊 RÉSULTATS TEST:")
            logger.info(f"🎤 Conversations: {len(pipeline.conversation_history)}")
            
            for i, conv in enumerate(pipeline.conversation_history[-3:], 1):  # 3 dernières
                logger.info(f"  {i}. Input: '{conv.user_input[:50]}...'")
                logger.info(f"     Réponse: '{conv.assistant_response[:50]}...'")
                logger.info(f"     Latence: {conv.total_latency:.1f}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Fonction principale"""
    try:
        success = await test_pipeline_reel()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Erreur main: {e}")
        return 1

if __name__ == "__main__":
    print("🎤 SuperWhisper V6 - Test Pipeline Réel Simple")
    print("🚨 Utilise le PipelineOrchestrator validé")
    print("🚨 Assurez-vous que:")
    print("  - Ollama server est démarré")
    print("  - Microphone RODE NT-USB est connecté")
    print("  - Speakers/casque sont connectés")
    print()
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 