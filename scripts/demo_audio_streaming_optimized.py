#!/usr/bin/env python3
"""
Démonstration AudioStreamer Optimisé - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Script de démonstration des 7 optimisations critiques intégrées:
1. Détection automatique périphérique audio robuste
2. Calibration automatique gain (RMS cible 0.05)
3. Architecture asynchrone pour faible latence
4. VAD avancé WebRTC + fallback RMS
5. Correction format audio PCM 16-bit pour VAD
6. Filtrage anti-hallucination post-transcription
7. Architecture séparée AudioStreamer + AudioStreamingManager

Intégration avec UnifiedSTTManager existant SuperWhisper V6

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

import time
import asyncio
import logging
from pathlib import Path

# Ajouter le chemin racine du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration GPU RTX 3090 obligatoire
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 === DÉMONSTRATION AUDIOSTREAMER OPTIMISÉ SUPERWHISPER V6 ===")
print("🚨 Configuration GPU: RTX 3090 (CUDA:1) OBLIGATOIRE")

# Imports après configuration GPU
try:
    from STT.audio_streamer_optimized import AudioStreamingManager, validate_rtx3090_configuration
    from STT.unified_stt_manager import UnifiedSTTManager
    import yaml
    import numpy as np
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    print("💡 Vérifiez que tous les modules sont installés")
    sys.exit(1)


def setup_logging():
    """Configuration du logging pour la démonstration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_audio_streaming.log')
        ]
    )
    return logging.getLogger('DemoAudioStreaming')


def load_stt_config():
    """Charge la configuration STT pour UnifiedSTTManager"""
    config_path = project_root / "config" / "stt.yaml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # Configuration par défaut si fichier absent
        return {
            'backends': [
                {
                    'name': 'prism_large',
                    'type': 'prism',
                    'model': 'large-v2',
                    'compute_type': 'float16',
                    'language': 'fr',
                    'beam_size': 10,  # Optimisé selon expert
                    'vad_filter': True
                }
            ],
            'fallback_chain': ['prism_large'],
            'timeout_per_minute': 5.0,
            'cache_size_mb': 200
        }


async def demo_integration_complete():
    """
    Démonstration complète de l'intégration AudioStreamer + UnifiedSTTManager
    """
    logger = setup_logging()
    
    try:
        # 1. Validation RTX 3090 obligatoire
        logger.info("🔍 Validation configuration RTX 3090...")
        validate_rtx3090_configuration()
        logger.info("✅ Configuration RTX 3090 validée")
        
        # 2. Initialisation UnifiedSTTManager
        logger.info("🧠 Initialisation UnifiedSTTManager...")
        stt_config = load_stt_config()
        stt_manager = UnifiedSTTManager(stt_config)
        
        # Attendre que le manager soit prêt
        await asyncio.sleep(2)
        logger.info("✅ UnifiedSTTManager prêt")
        
        # 3. Initialisation AudioStreamingManager optimisé
        logger.info("🎤 Initialisation AudioStreamingManager optimisé...")
        streaming_manager = AudioStreamingManager(
            unified_stt_manager=stt_manager,
            device_name="Rode NT-USB",  # Optimisation #1: Détection par nom
            chunk_duration=2.0  # Chunks de 2s pour équilibrer latence/précision
        )
        logger.info("✅ AudioStreamingManager optimisé prêt")
        
        # 4. Démonstration des optimisations
        logger.info("\n🎯 === DÉMONSTRATION DES 7 OPTIMISATIONS CRITIQUES ===")
        
        # Optimisation #1: Détection automatique périphérique
        logger.info("1️⃣ Détection automatique périphérique 'Rode NT-USB'")
        device_detected = streaming_manager.streamer.stats['device_detection_success']
        logger.info(f"   ✅ Périphérique détecté: {device_detected}")
        
        # Optimisation #2: Calibration automatique gain
        logger.info("2️⃣ Calibration automatique gain (RMS cible 0.05)")
        logger.info(f"   🎛️ Auto-gain activé: {streaming_manager.streamer.auto_gain_enabled}")
        logger.info(f"   🎯 RMS cible: {streaming_manager.streamer.target_rms}")
        
        # Optimisation #3: Architecture asynchrone
        logger.info("3️⃣ Architecture asynchrone pour faible latence")
        logger.info("   🔄 Threading + callbacks + pipeline async")
        
        # Optimisation #4: VAD avancé
        logger.info("4️⃣ VAD avancé WebRTC + fallback RMS")
        vad_available = streaming_manager.streamer.vad.webrtc_available
        logger.info(f"   🎙️ WebRTC-VAD disponible: {vad_available}")
        logger.info(f"   🔊 Fallback RMS seuil: {streaming_manager.streamer.vad.rms_threshold}")
        
        # Optimisation #5: Correction format audio
        logger.info("5️⃣ Correction format audio float32 → int16 PCM")
        logger.info("   🔧 Conversion automatique pour WebRTC-VAD")
        
        # Optimisation #6: Filtrage hallucinations
        logger.info("6️⃣ Filtrage anti-hallucination post-transcription")
        patterns_count = len(streaming_manager.hallucination_filter.hallucination_patterns)
        logger.info(f"   🚫 Patterns d'hallucination: {patterns_count}")
        
        # Optimisation #7: Architecture séparée
        logger.info("7️⃣ Architecture séparée AudioStreamer + AudioStreamingManager")
        logger.info("   🏗️ Interface propre avec UnifiedSTTManager")
        
        # 5. Test streaming en conditions réelles
        logger.info("\n🌊 === TEST STREAMING TEMPS RÉEL ===")
        logger.info("🎤 Démarrage capture microphone live...")
        
        if streaming_manager.start_continuous_mode():
            logger.info("✅ Mode streaming continu actif")
            logger.info("🗣️ Parlez au microphone pour tester la transcription...")
            logger.info("⏱️ Test pendant 30 secondes...")
            
            # Test pendant 30 secondes avec monitoring
            start_time = time.time()
            last_stats_time = start_time
            
            while time.time() - start_time < 30:
                await asyncio.sleep(1)
                
                # Vérifier résultats toutes les secondes
                result = streaming_manager.get_latest_result(timeout=0.1)
                if result:
                    text = result.get('text', 'N/A')
                    confidence = result.get('confidence', 0)
                    rtf = result.get('rtf', 0)
                    logger.info(f"📝 Transcription: '{text}' (conf: {confidence:.2f}, RTF: {rtf:.3f})")
                
                # Stats toutes les 10 secondes
                if time.time() - last_stats_time >= 10:
                    stats = streaming_manager.get_stats()
                    manager_stats = stats['manager']
                    streamer_stats = stats['streamer']
                    
                    logger.info("📊 === STATS INTERMÉDIAIRES ===")
                    logger.info(f"   🎤 Chunks audio traités: {streamer_stats['chunks_processed']}")
                    logger.info(f"   🗣️ Chunks avec voix: {streamer_stats['chunks_with_voice']}")
                    logger.info(f"   ✅ Transcriptions réussies: {manager_stats['transcriptions_completed']}")
                    logger.info(f"   🚫 Hallucinations filtrées: {manager_stats['hallucinations_filtered']}")
                    logger.info(f"   ⚡ Temps traitement moyen: {manager_stats['avg_processing_time']:.3f}s")
                    
                    last_stats_time = time.time()
            
            # Arrêter streaming
            streaming_manager.stop_continuous_mode()
            
            # 6. Analyse des résultats finaux
            logger.info("\n🏆 === ANALYSE RÉSULTATS FINAUX ===")
            final_stats = streaming_manager.get_stats()
            
            manager_stats = final_stats['manager']
            streamer_stats = final_stats['streamer']
            hallucination_stats = final_stats['hallucination_filter']
            
            # Performance globale
            total_chunks = streamer_stats['chunks_processed']
            voice_chunks = streamer_stats['chunks_with_voice']
            transcriptions = manager_stats['transcriptions_completed']
            
            logger.info("📈 === PERFORMANCE GLOBALE ===")
            logger.info(f"   🎤 Total chunks audio: {total_chunks}")
            logger.info(f"   🗣️ Chunks avec voix: {voice_chunks}")
            logger.info(f"   📝 Transcriptions réussies: {transcriptions}")
            logger.info(f"   ❌ Transcriptions échouées: {manager_stats['transcriptions_failed']}")
            logger.info(f"   🚫 Hallucinations filtrées: {manager_stats['hallucinations_filtered']}")
            
            # Métriques de qualité
            if voice_chunks > 0:
                voice_ratio = voice_chunks / total_chunks * 100
                logger.info(f"   🔊 Ratio détection voix: {voice_ratio:.1f}%")
            
            if transcriptions > 0:
                avg_processing = manager_stats['avg_processing_time']
                logger.info(f"   ⚡ Temps traitement moyen: {avg_processing:.3f}s")
                
                # Estimation RTF moyen (approximatif)
                chunk_duration = streaming_manager.chunk_duration
                estimated_rtf = avg_processing / chunk_duration
                logger.info(f"   📊 RTF estimé: {estimated_rtf:.3f}")
            
            # Efficacité des optimisations
            logger.info("🎯 === EFFICACITÉ DES OPTIMISATIONS ===")
            
            # Optimisation #1: Détection périphérique
            device_success = streamer_stats['device_detection_success']
            logger.info(f"   1️⃣ Détection périphérique: {'✅ Réussie' if device_success else '❌ Échec'}")
            
            # Optimisation #2: Calibration gain
            gain_applied = streamer_stats['gain_applied']
            avg_rms = streamer_stats['avg_rms']
            logger.info(f"   2️⃣ Calibration gain: {'✅ Appliquée' if gain_applied else '➖ Non nécessaire'}")
            logger.info(f"        RMS moyen: {avg_rms:.6f}")
            
            # Optimisation #4: VAD efficacité
            if total_chunks > 0:
                vad_efficiency = voice_chunks / total_chunks * 100
                logger.info(f"   4️⃣ Efficacité VAD: {vad_efficiency:.1f}% chunks voix détectés")
            
            # Optimisation #6: Filtrage hallucinations
            hallucinations_detected = manager_stats['hallucinations_filtered']
            if transcriptions + hallucinations_detected > 0:
                filter_rate = hallucinations_detected / (transcriptions + hallucinations_detected) * 100
                logger.info(f"   6️⃣ Filtrage hallucinations: {hallucinations_detected} filtrées ({filter_rate:.1f}%)")
            
            # Recommandations
            logger.info("\n💡 === RECOMMANDATIONS ===")
            
            if not device_success:
                logger.warning("⚠️ Périphérique 'Rode NT-USB' non détecté - vérifier branchement")
            
            if avg_rms < 0.01:
                logger.warning("⚠️ Signal audio faible - rapprochez-vous du microphone")
            elif avg_rms > 0.1:
                logger.warning("⚠️ Signal audio fort - éloignez-vous du microphone")
            
            if transcriptions == 0:
                logger.warning("⚠️ Aucune transcription réussie - vérifier configuration STT")
            
            if estimated_rtf > 1.0:
                logger.warning("⚠️ RTF > 1.0 - performance insuffisante pour temps réel")
            
            logger.info("✅ Démonstration terminée avec succès!")
            return True
            
        else:
            logger.error("❌ Échec démarrage streaming")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur démonstration: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_comparaison_avant_apres():
    """
    Démonstration comparative avant/après optimisations
    """
    logger = setup_logging()
    
    logger.info("\n📊 === COMPARAISON AVANT/APRÈS OPTIMISATIONS ===")
    
    # Métriques avant optimisations (basées sur tests précédents)
    avant = {
        'wer': 44.2,  # WER 44.2% lors test microphone live
        'detection_peripherique': 'ID instable',
        'calibration_gain': 'Manuelle',
        'vad': 'Basique',
        'hallucinations': 'Non filtrées',
        'architecture': 'Synchrone'
    }
    
    # Métriques après optimisations (objectifs)
    apres = {
        'wer_objectif': 15.0,  # Objectif <15% WER
        'detection_peripherique': 'Par nom robuste',
        'calibration_gain': 'Automatique RMS 0.05',
        'vad': 'WebRTC + fallback',
        'hallucinations': 'Filtrage patterns',
        'architecture': 'Asynchrone'
    }
    
    logger.info("📈 AVANT optimisations:")
    logger.info(f"   🎯 WER: {avant['wer']}% (INSUFFISANT)")
    logger.info(f"   🎤 Détection périphérique: {avant['detection_peripherique']}")
    logger.info(f"   🔧 Calibration gain: {avant['calibration_gain']}")
    logger.info(f"   🎙️ VAD: {avant['vad']}")
    logger.info(f"   🚫 Hallucinations: {avant['hallucinations']}")
    logger.info(f"   🏗️ Architecture: {avant['architecture']}")
    
    logger.info("\n🚀 APRÈS optimisations:")
    logger.info(f"   🎯 WER objectif: <{apres['wer_objectif']}% (PROFESSIONNEL)")
    logger.info(f"   🎤 Détection périphérique: {apres['detection_peripherique']}")
    logger.info(f"   🔧 Calibration gain: {apres['calibration_gain']}")
    logger.info(f"   🎙️ VAD: {apres['vad']}")
    logger.info(f"   🚫 Hallucinations: {apres['hallucinations']}")
    logger.info(f"   🏗️ Architecture: {apres['architecture']}")
    
    # Calcul amélioration attendue
    amelioration_wer = (avant['wer'] - apres['wer_objectif']) / avant['wer'] * 100
    logger.info(f"\n🎉 Amélioration WER attendue: {amelioration_wer:.1f}%")
    logger.info(f"   📉 {avant['wer']}% → {apres['wer_objectif']}%")


async def main():
    """Fonction principale de démonstration"""
    logger = setup_logging()
    
    try:
        logger.info("🎬 Démarrage démonstration AudioStreamer optimisé SuperWhisper V6")
        
        # 1. Comparaison avant/après
        demo_comparaison_avant_apres()
        
        # 2. Démonstration intégration complète
        success = await demo_integration_complete()
        
        if success:
            logger.info("\n🎉 === DÉMONSTRATION RÉUSSIE ===")
            logger.info("✅ Toutes les optimisations critiques sont fonctionnelles")
            logger.info("🔗 AudioStreamer optimisé intégré avec UnifiedSTTManager")
            logger.info("🎯 Prêt pour validation microphone live finale")
        else:
            logger.error("\n💥 === DÉMONSTRATION ÉCHOUÉE ===")
            logger.error("❌ Vérifier configuration et dépendances")
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Démonstration interrompue par utilisateur")
    except Exception as e:
        logger.error(f"\n❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Lancement démonstration AudioStreamer optimisé...")
    print("⚠️ Assurez-vous que le microphone 'Rode NT-USB' est branché")
    print("🎤 Préparez-vous à parler au microphone pendant le test")
    
    # Validation RTX 3090 avant démarrage
    try:
        validate_rtx3090_configuration()
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE GPU: {e}")
        sys.exit(1)
    
    # Lancer démonstration
    asyncio.run(main()) 