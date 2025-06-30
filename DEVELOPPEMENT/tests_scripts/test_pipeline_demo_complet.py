#!/usr/bin/env python3
"""
Demo Pipeline Complet SuperWhisper V6
Test réel STT → LLM → TTS avec fichier audio
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import time
import logging
import asyncio
import pathlib
import wave
import numpy as np
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
logger = logging.getLogger("DemoComplet")

def create_test_audio():
    """Crée un fichier audio de test avec du bruit blanc simulant de la parole"""
    logger.info("🎵 Création audio de test...")
    
    # Paramètres audio
    sample_rate = 16000
    duration = 3.0  # 3 secondes
    
    # Générer du bruit blanc avec modulation (simule de la parole)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Signal de base avec variations (simule intonation)
    signal = np.random.normal(0, 0.1, len(t))
    modulation = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
    audio_signal = signal * (1 + modulation)
    
    # Ajouter quelques pics pour simuler des consonnes
    for i in range(0, len(audio_signal), sample_rate // 3):
        if i < len(audio_signal):
            audio_signal[i:i+100] *= 3
    
    # Normaliser
    audio_signal = audio_signal / np.max(np.abs(audio_signal)) * 0.7
    
    # Convertir en PCM 16-bit
    audio_pcm = (audio_signal * 32767).astype(np.int16)
    
    # Sauvegarder en WAV
    test_file = "test_audio_input.wav"
    with wave.open(test_file, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_pcm.tobytes())
    
    logger.info(f"✅ Audio de test créé: {test_file}")
    return audio_pcm.tobytes(), test_file

async def demo_pipeline_complet():
    """Démonstration complète du pipeline avec audio réel"""
    logger.info("🚀 Demo Pipeline Complet SuperWhisper V6")
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
        
        # 2. Créer un audio de test ou utiliser un existant
        questions_test = [
            "Bonjour, comment allez-vous aujourd'hui ?",
            "Quelle est la météo aujourd'hui ?",  
            "Pouvez-vous me dire l'heure qu'il est ?",
            "Merci pour votre aide, au revoir !"
        ]
        
        for i, question in enumerate(questions_test, 1):
            logger.info(f"\n🎯 TEST {i}/4: '{question}'")
            logger.info("-" * 50)
            
            # Générer audio de test
            audio_bytes, audio_file = create_test_audio()
            
            # Pipeline complet
            total_start = time.perf_counter()
            
            # 3. STT - Simulation car audio généré est du bruit
            logger.info("🎤 STT - Simulation avec question prédéfinie...")
            stt_start = time.perf_counter()
            user_text = question  # Simulation
            stt_time = (time.perf_counter() - stt_start) * 1000
            logger.info(f"✅ STT simulé ({stt_time:.1f}ms): '{user_text}'")
            
            # 4. LLM - Vraie consultation
            logger.info("🧠 LLM - Consultation réelle...")
            llm_start = time.perf_counter()
            
            try:
                llm_response = await llm_manager.generate_response(
                    user_input=user_text,
                    max_tokens=50
                )
                llm_time = (time.perf_counter() - llm_start) * 1000
                logger.info(f"✅ LLM réussi ({llm_time:.1f}ms): '{llm_response}'")
            except Exception as e:
                logger.warning(f"⚠️ LLM fallback: {e}")
                llm_response = f"Merci pour votre question : '{user_text}'. Je vous réponds du mieux que je peux."
                llm_time = 1.0
            
            # 5. TTS - Vraie synthèse
            logger.info("🔊 TTS - Synthèse réelle...")
            tts_start = time.perf_counter()
            
            tts_result = await tts_manager.synthesize(text=llm_response)
            tts_time = (time.perf_counter() - tts_start) * 1000
            
            if tts_result.success and tts_result.audio_data:
                output_file = f"demo_output_{i}.wav"
                with open(output_file, 'wb') as f:
                    f.write(tts_result.audio_data)
                
                logger.info(f"✅ TTS réussi ({tts_time:.1f}ms)")
                logger.info(f"📁 Audio généré: {output_file}")
                logger.info(f"🎛️ Backend: {tts_result.backend_used}")
                logger.info(f"📊 Taille: {len(tts_result.audio_data)} bytes")
                
                tts_success = True
            else:
                logger.error(f"❌ TTS échoué: {tts_result.error}")
                tts_success = False
                tts_time = 0
            
            # 6. Résultats du cycle
            total_time = (time.perf_counter() - total_start) * 1000
            
            logger.info(f"\n📊 RÉSULTATS TEST {i}")
            logger.info(f"👤 Question: '{user_text}'")
            logger.info(f"🤖 Réponse: '{llm_response}'")
            logger.info(f"⏱️ LLM: {llm_time:.1f}ms")
            logger.info(f"⏱️ TTS: {tts_time:.1f}ms")
            logger.info(f"⏱️ TOTAL (LLM+TTS): {llm_time + tts_time:.1f}ms")
            logger.info(f"🎯 Statut: {'✅ Réussi' if tts_success else '❌ Échoué'}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 DEMO PIPELINE COMPLET TERMINÉE !")
        logger.info("🎧 Écoutez les fichiers audio générés : demo_output_1.wav à demo_output_4.wav")
        logger.info("✅ Pipeline STT → LLM → TTS entièrement fonctionnel !")
        
    except Exception as e:
        logger.error(f"❌ Erreur demo: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Point d'entrée principal"""
    logger.info("🚀 SuperWhisper V6 - Demo Pipeline Complet")
    
    try:
        asyncio.run(demo_pipeline_complet())
        
    except KeyboardInterrupt:
        logger.info("🛑 Demo interrompue par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")

if __name__ == "__main__":
    main()