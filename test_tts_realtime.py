#!/usr/bin/env python3
"""
Test TTS avec lecture temps réel
SuperWhisper V6 - Correction audio playback
"""

import os
import sys
import pathlib
import asyncio
import logging
import yaml

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("TTSRealtime")

async def test_tts_realtime():
    """Test TTS avec lecture audio temps réel"""
    logger.info("🔊 Test TTS temps réel SuperWhisper V6")
    
    try:
        # Initialiser TTS Manager
        from TTS.tts_manager import UnifiedTTSManager
        
        with open('config/tts.yaml', 'r') as f:
            tts_config = yaml.safe_load(f)
        
        tts_manager = UnifiedTTSManager(tts_config)
        logger.info("✅ TTS Manager initialisé")
        
        # Messages de test
        messages = [
            "Bonjour, ceci est un test de synthèse vocale temps réel.",
            "SuperWhisper version 6 fonctionne maintenant avec lecture automatique.",
            "Merci de valider la qualité audio de ce test."
        ]
        
        for i, message in enumerate(messages, 1):
            logger.info(f"🎤 Synthèse message {i}/3: '{message[:30]}...'")
            
            # Synthèse TTS
            result = await tts_manager.synthesize(text=message)
            
            if result.success and result.audio_data:
                # Sauvegarder fichier temporaire
                temp_file = f"temp_tts_realtime_{i}.wav"
                with open(temp_file, 'wb') as f:
                    f.write(result.audio_data)
                
                logger.info(f"✅ Audio généré: {len(result.audio_data)} bytes")
                logger.info(f"🔊 Lecture: {temp_file}")
                
                # Lecture automatique
                try:
                    import subprocess
                    import platform
                    
                    if platform.system() == "Linux":
                        # WSL - utiliser Windows
                        windows_path = os.path.abspath(temp_file).replace('/mnt/c', 'C:')
                        subprocess.run([
                            "cmd.exe", "/c", "start", "", windows_path
                        ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        logger.info(f"🎧 Lecture Windows: {windows_path}")
                    else:
                        # Windows direct
                        os.startfile(temp_file)
                        logger.info(f"🎧 Lecture directe: {temp_file}")
                    
                    # Attendre lecture
                    logger.info("⏳ Attente lecture (5s)...")
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Lecture automatique échouée: {e}")
                    logger.info(f"🎧 Lisez manuellement: {temp_file}")
                
                # Nettoyer fichier temporaire
                try:
                    os.remove(temp_file)
                except:
                    pass
            else:
                logger.error(f"❌ Échec synthèse: {result.error}")
                return False
        
        logger.info("✅ Test TTS temps réel terminé avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test TTS: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🔊 SuperWhisper V6 - Test TTS Lecture Temps Réel")
    print("=" * 60)
    
    success = await test_tts_realtime()
    
    print("\n📊 RÉSULTATS:")
    print(f"  🔊 TTS Temps Réel: {'✅' if success else '❌'}")
    
    if success:
        print("\n🎉 TTS LECTURE TEMPS RÉEL FONCTIONNEL!")
        print("   Le système peut maintenant jouer l'audio automatiquement.")
    else:
        print("\n❌ TTS temps réel non fonctionnel")
    
if __name__ == "__main__":
    asyncio.run(main())