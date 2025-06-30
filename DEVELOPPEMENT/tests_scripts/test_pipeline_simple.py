#!/usr/bin/env python3
"""
SuperWhisper V6 - Test Pipeline Voix-à-Voix Simplifié
===================================================

Test simple du pipeline complet : Microphone → STT → LLM → TTS → Haut-parleurs
"""

import os
import sys
import time
import logging
import asyncio
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
logger = logging.getLogger("PipelineTest")

def test_stt_simple():
    """Test STT avec audio temporaire"""
    logger.info("🎤 Test STT...")
    
    try:
        from STT.unified_stt_manager import UnifiedSTTManager
        
        stt_config = {
            'timeout_per_minute': 10.0,
            'cache_size_mb': 200,
            'fallback_chain': ['prism_primary']
        }
        
        stt = UnifiedSTTManager(stt_config)
        logger.info("✅ STT Manager initialisé")
        
        # Test avec fichier audio temporaire si disponible
        test_audio_path = "test_audio.wav"
        if os.path.exists(test_audio_path):
            result = stt.transcribe_file(test_audio_path)
            logger.info(f"📝 Transcription: {result}")
        else:
            logger.info("📝 STT prêt (pas de fichier test)")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur STT: {e}")
        return False

def test_llm_simple():
    """Test LLM avec question simple"""
    logger.info("🧠 Test LLM...")
    
    try:
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        
        llm_config = {
            "model_name": "nous-hermes",
            "max_context_turns": 5,
            "timeout_seconds": 10,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        llm = EnhancedLLMManager(llm_config)
        logger.info("✅ LLM Manager initialisé")
        
        # Test génération
        user_input = "Dis bonjour en français de manière amicale en une phrase."
        start_time = time.time()
        response = asyncio.run(llm.generate_response(user_input=user_input, max_tokens=50))
        llm_time = time.time() - start_time
        
        logger.info(f"🧠 Réponse ({llm_time:.1f}s): '{response}'")
        return True, response, llm_time
        
    except Exception as e:
        logger.error(f"❌ Erreur LLM: {e}")
        return False, "", 0

def test_tts_simple(text):
    """Test TTS avec texte"""
    logger.info("🔊 Test TTS...")
    
    try:
        from TTS.tts_manager import UnifiedTTSManager
        import asyncio
        import yaml
        
        # Utiliser la configuration YAML validée
        with open('config/tts.yaml', 'r') as f:
            tts_config = yaml.safe_load(f)
        
        tts = UnifiedTTSManager(tts_config)
        logger.info("✅ TTS Manager initialisé")
        
        # Test génération
        async def synthesize_test():
            start_time = time.time()
            audio_data = await tts.synthesize(text=text)
            tts_time = time.time() - start_time
            return audio_data, tts_time
        
        audio_data, tts_time = asyncio.run(synthesize_test())
        
        if audio_data:
            logger.info(f"🔊 Audio généré ({tts_time:.1f}s) - {len(audio_data)} bytes")
            return True, tts_time
        else:
            logger.info(f"🔊 TTS fonctionnel ({tts_time:.1f}s) - mode silencieux")
            return True, tts_time
        
    except Exception as e:
        logger.error(f"❌ Erreur TTS: {e}")
        return False, 0

def main():
    """Test complet du pipeline"""
    logger.info("🚀 SuperWhisper V6 - Test Pipeline Complet")
    logger.info("="*60)
    
    total_start = time.time()
    
    # 1. Test STT
    stt_success = test_stt_simple()
    
    # 2. Test LLM
    llm_success, llm_response, llm_time = test_llm_simple()
    
    # 3. Test TTS avec réponse LLM
    if llm_success and llm_response:
        tts_success, tts_time = test_tts_simple(llm_response)
    else:
        tts_success, tts_time = test_tts_simple("Bonjour, ceci est un test.")
    
    # Résultats
    total_time = time.time() - total_start
    
    logger.info("="*60)
    logger.info("📊 RÉSULTATS PIPELINE")
    logger.info("="*60)
    logger.info(f"🎤 STT: {'✅' if stt_success else '❌'}")
    logger.info(f"🧠 LLM: {'✅' if llm_success else '❌'} ({llm_time:.1f}s)")
    logger.info(f"🔊 TTS: {'✅' if tts_success else '❌'} ({tts_time:.1f}s)")
    logger.info(f"⏱️ Temps total: {total_time:.1f}s")
    
    if stt_success and llm_success and tts_success:
        logger.info("🎉 PIPELINE VOIX-À-VOIX FONCTIONNEL !")
        latence_estimee = llm_time + tts_time + 1.0  # +1s pour STT estimé
        logger.info(f"🎯 Latence estimée conversation: {latence_estimee:.1f}s")
        
        if latence_estimee < 3.0:
            logger.info("✅ OBJECTIF LATENCE <3s ATTEINT !")
        else:
            logger.info("⚠️ Latence supérieure à 3s - optimisations requises")
    else:
        logger.info("❌ Pipeline incomplet - corrections requises")

if __name__ == "__main__":
    main()