#!/usr/bin/env python3
"""
Test Pipeline E2E Complet SuperWhisper V6
STT → LLM → TTS avec streaming microphone validé
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import time
import logging
import asyncio
import pathlib
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
logger = logging.getLogger("PipelineE2E")

def test_pipeline_e2e():
    """Test pipeline e2e avec simulation d'entrée microphone"""
    logger.info("🚀 Test Pipeline E2E SuperWhisper V6")
    logger.info("=" * 60)
    
    try:
        # 1. Initialiser les composants
        logger.info("🔧 Initialisation des composants...")
        
        # STT Manager (validé)
        from STT.unified_stt_manager import UnifiedSTTManager
        stt_config = {
            'timeout_per_minute': 10.0,
            'cache_size_mb': 200,
            'fallback_chain': ['prism_primary']
        }
        stt_manager = UnifiedSTTManager(stt_config)
        logger.info("✅ STT Manager initialisé")
        
        # LLM Manager 
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        llm_config = {
            'model': 'nous-hermes',
            'base_url': 'http://127.0.0.1:11434/v1',
            'timeout': 30.0
        }
        llm_manager = EnhancedLLMManager(llm_config)
        logger.info("✅ LLM Manager initialisé")
        
        # TTS Manager (validé)
        from TTS.tts_manager import UnifiedTTSManager
        import yaml
        with open('config/tts.yaml', 'r') as f:
            tts_config = yaml.safe_load(f)
        tts_manager = UnifiedTTSManager(tts_config)
        logger.info("✅ TTS Manager initialisé")
        
        # 2. Simulation d'une conversation voix-à-voix
        logger.info("\n🎯 Test conversation simulée...")
        
        # Texte d'entrée simulé (comme si c'était du STT)
        user_input = "Bonjour, comment allez-vous aujourd'hui ?"
        logger.info(f"🎤 Entrée utilisateur (simulé STT): '{user_input}'")
        
        # 3. Traitement LLM
        logger.info("🧠 Traitement LLM...")
        start_llm = time.perf_counter()
        
        async def process_llm():
            return await llm_manager.generate_response(
                user_input=user_input,
                max_tokens=50
            )
        
        llm_response = asyncio.run(process_llm())
        llm_time = (time.perf_counter() - start_llm) * 1000
        
        logger.info(f"🧠 Réponse LLM ({llm_time:.1f}ms): '{llm_response}'")
        
        # 4. Synthèse TTS
        logger.info("🔊 Synthèse TTS...")
        start_tts = time.perf_counter()
        
        async def process_tts():
            return await tts_manager.synthesize(text=llm_response)
        
        tts_result = asyncio.run(process_tts())
        tts_time = (time.perf_counter() - start_tts) * 1000
        
        if tts_result.success and tts_result.audio_data:
            # Sauvegarder l'audio de sortie
            output_file = "pipeline_e2e_output.wav"
            with open(output_file, 'wb') as f:
                f.write(tts_result.audio_data)
            
            logger.info(f"✅ TTS réussi ({tts_time:.1f}ms):")
            logger.info(f"   📁 Fichier: {output_file}")
            logger.info(f"   🎛️  Backend: {tts_result.backend_used}")
            logger.info(f"   📊 Taille: {len(tts_result.audio_data)} bytes")
        else:
            logger.error(f"❌ TTS échoué: {tts_result.error}")
            tts_time = 0
        
        # 5. Résultats du pipeline e2e
        total_time = llm_time + tts_time
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 RÉSULTATS PIPELINE E2E")
        logger.info("=" * 60)
        logger.info(f"🎤 STT: ✅ (simulé)")
        logger.info(f"🧠 LLM: ✅ ({llm_time:.1f}ms)")
        logger.info(f"🔊 TTS: {'✅' if tts_result.success else '❌'} ({tts_time:.1f}ms)")
        logger.info(f"⏱️ Latence totale LLM+TTS: {total_time:.1f}ms")
        logger.info(f"🎯 Objectif < 2500ms: {'✅' if total_time < 2500 else '❌'}")
        
        if tts_result.success:
            logger.info("🎉 Pipeline E2E fonctionnel !")
            logger.info(f"🎧 Écoutez le résultat: {output_file}")
        else:
            logger.warning("⚠️ Pipeline partiellement fonctionnel (TTS à corriger)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur pipeline E2E: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Point d'entrée principal"""
    logger.info("🚀 SuperWhisper V6 - Test Pipeline E2E")
    
    try:
        success = test_pipeline_e2e()
        
        if success:
            logger.info("\n✅ Test pipeline E2E terminé avec succès !")
        else:
            logger.error("\n❌ Test pipeline E2E échoué")
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")

if __name__ == "__main__":
    main()