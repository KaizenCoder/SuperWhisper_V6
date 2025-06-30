#!/usr/bin/env python3
"""
Test d'écoute TTS SuperWhisper V6
Synthèse vocale française avec lecture audio
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
logger = logging.getLogger("TTSTest")

async def test_tts_ecoute():
    """Test d'écoute TTS avec audio en sortie"""
    logger.info("🔊 Test TTS SuperWhisper V6 - Écoute audio")
    logger.info("=" * 60)
    
    try:
        # Import des modules TTS
        from TTS.tts_manager import UnifiedTTSManager
        import yaml
        
        # Charger la configuration TTS validée
        logger.info("📋 Chargement configuration TTS...")
        with open('config/tts.yaml', 'r') as f:
            tts_config = yaml.safe_load(f)
        
        # Initialiser le TTS Manager
        logger.info("🚀 Initialisation UnifiedTTSManager...")
        tts = UnifiedTTSManager(tts_config)
        logger.info("✅ TTS Manager initialisé avec succès")
        
        # Textes de test
        textes_test = [
            "Bonjour, je suis l'assistant vocal SuperWhisper version 6.",
            "Test de qualité audio avec voix française naturelle.",
            "Les performances sont optimisées pour RTX 3090.",
            "Merci de valider la qualité sonore de cette synthèse vocale."
        ]
        
        logger.info("🎤 Génération des échantillons audio...")
        
        for i, texte in enumerate(textes_test, 1):
            logger.info(f"\n📝 Test {i}/4: '{texte}'")
            
            # Mesure de performance
            start_time = time.perf_counter()
            
            # Synthèse vocale
            result = await tts.synthesize(text=texte)
            
            end_time = time.perf_counter()
            latence_ms = (end_time - start_time) * 1000
            
            if result.success and result.audio_data:
                # Sauvegarder l'audio
                output_file = f"test_tts_output_{i}.wav"
                with open(output_file, 'wb') as f:
                    f.write(result.audio_data)
                
                logger.info(f"✅ Synthèse réussie:")
                logger.info(f"   📁 Fichier: {output_file}")
                logger.info(f"   ⏱️  Latence: {latence_ms:.1f}ms")
                logger.info(f"   🎛️  Backend: {result.backend_used}")
                logger.info(f"   📊 Taille: {len(result.audio_data)} bytes")
                
                # Lecture audio automatique (Windows)
                try:
                    import subprocess
                    logger.info(f"🔊 Lecture audio automatique...")
                    # Utiliser le lecteur audio par défaut de Windows
                    subprocess.Popen([
                        "powershell", "-c", 
                        f"Add-Type -AssemblyName System.Speech; "
                        f"(New-Object System.Media.SoundPlayer('{os.path.abspath(output_file)}')).PlaySync()"
                    ], shell=True)
                except Exception as e:
                    logger.warning(f"⚠️ Lecture auto impossible: {e}")
                    logger.info(f"🎧 Lisez manuellement: {output_file}")
                
                # Pause entre les tests
                if i < len(textes_test):
                    logger.info("⏳ Pause 2s avant le prochain test...")
                    await asyncio.sleep(2)
                    
            else:
                logger.error(f"❌ Échec synthèse: {result.error}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Test TTS terminé!")
        logger.info("🎧 Vérifiez la qualité audio des fichiers générés")
        logger.info("📁 Fichiers: test_tts_output_1.wav à test_tts_output_4.wav")
        
    except Exception as e:
        logger.error(f"❌ Erreur test TTS: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Point d'entrée principal"""
    logger.info("🚀 SuperWhisper V6 - Test écoute TTS")
    
    try:
        # Lancer le test async
        asyncio.run(test_tts_ecoute())
        
    except KeyboardInterrupt:
        logger.info("🛑 Test interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")

if __name__ == "__main__":
    main()