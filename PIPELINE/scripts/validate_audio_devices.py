#!/usr/bin/env python3
"""
🎤 PRÉ-FLIGHT CHECK PIPELINE - VALIDATION AUDIO DEVICES WINDOWS
==============================================================
Script de validation des devices audio pour pipeline SuperWhisper V6

VALIDATIONS :
- Permissions Windows audio (microphone + speakers)
- Énumération devices disponibles  
- Test capture/playback basique
- Configuration format audio optimal

Usage: python PIPELINE/scripts/validate_audio_devices.py

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

import logging
import platform
from typing import Dict, List, Any, Optional

# Configuration logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate_audio_devices")

def validate_windows_audio_permissions() -> Dict[str, Any]:
    """
    Validation des permissions audio Windows pour pipeline
    
    Returns:
        Dict contenant les résultats de validation
    """
    validation_results = {
        'platform': platform.system(),
        'python_version': sys.version,
        'available_apis': [],
        'input_devices': [],
        'output_devices': [],
        'default_input': None,
        'default_output': None,
        'permissions_ok': False
    }
    
    try:
        # 1. Vérification plateforme
        if platform.system() != 'Windows':
            logger.warning(f"⚠️ Plateforme {platform.system()} - optimisé pour Windows")
        
        # 2. Import PyAudio pour énumération devices
        try:
            import pyaudio
            validation_results['available_apis'].append('PyAudio')
            logger.info("✅ PyAudio disponible")
        except ImportError:
            logger.error("❌ PyAudio non disponible - installer avec: pip install pyaudio")
            raise RuntimeError("PyAudio requis pour validation audio")
        
        # 3. Énumération devices audio
        pa = pyaudio.PyAudio()
        
        try:
            # Devices d'entrée (microphones)
            input_devices = []
            output_devices = []
            
            for i in range(pa.get_device_count()):
                device_info = pa.get_device_info_by_index(i)
                
                if device_info['maxInputChannels'] > 0:
                    input_device = {
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate']),
                        'api': pa.get_host_api_info_by_index(device_info['hostApi'])['name']
                    }
                    input_devices.append(input_device)
                    
                if device_info['maxOutputChannels'] > 0:
                    output_device = {
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxOutputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate']),
                        'api': pa.get_host_api_info_by_index(device_info['hostApi'])['name']
                    }
                    output_devices.append(output_device)
            
            validation_results['input_devices'] = input_devices
            validation_results['output_devices'] = output_devices
            
            # Device par défaut
            try:
                default_input_info = pa.get_default_input_device_info()
                validation_results['default_input'] = {
                    'index': default_input_info['index'],
                    'name': default_input_info['name'],
                    'channels': default_input_info['maxInputChannels'],
                    'sample_rate': int(default_input_info['defaultSampleRate'])
                }
            except Exception as e:
                logger.warning(f"⚠️ Aucun device d'entrée par défaut: {e}")
            
            try:
                default_output_info = pa.get_default_output_device_info()
                validation_results['default_output'] = {
                    'index': default_output_info['index'],
                    'name': default_output_info['name'],
                    'channels': default_output_info['maxOutputChannels'],
                    'sample_rate': int(default_output_info['defaultSampleRate'])
                }
            except Exception as e:
                logger.warning(f"⚠️ Aucun device de sortie par défaut: {e}")
                
        finally:
            pa.terminate()
        
        # 4. Validation résultats
        if len(input_devices) == 0:
            raise RuntimeError("❌ Aucun device d'entrée (microphone) détecté")
        
        if len(output_devices) == 0:
            raise RuntimeError("❌ Aucun device de sortie (speakers) détecté")
        
        if validation_results['default_input'] is None:
            logger.warning("⚠️ Aucun microphone par défaut configuré")
        
        validation_results['permissions_ok'] = True
        logger.info(f"✅ Devices audio validés: {len(input_devices)} entrée(s), {len(output_devices)} sortie(s)")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ Échec validation audio: {e}")
        validation_results['error'] = str(e)
        raise

def test_audio_capture_basic() -> bool:
    """
    Test basique de capture audio pour validation permissions
    
    Returns:
        True si capture fonctionne
    """
    try:
        import pyaudio
        import numpy as np
        
        # Configuration test
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        TEST_DURATION = 0.1  # 100ms
        
        logger.info("🎤 Test capture audio basique...")
        
        pa = pyaudio.PyAudio()
        
        try:
            # Ouverture stream capture
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            # Capture courte
            frames = []
            for i in range(int(RATE / CHUNK * TEST_DURATION)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # Vérification signal
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Statistiques basiques
            rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
            max_amplitude = np.max(np.abs(audio_array))
            
            logger.info(f"✅ Capture test réussie - RMS: {rms:.1f}, Max: {max_amplitude}")
            
            return True
            
        finally:
            pa.terminate()
            
    except Exception as e:
        logger.error(f"❌ Échec test capture: {e}")
        return False

def main():
    """Point d'entrée principal du script de validation audio"""
    logger.info("🎤 DÉMARRAGE VALIDATION AUDIO DEVICES PIPELINE...")
    
    try:
        # 1. Validation devices et permissions
        results = validate_windows_audio_permissions()
        
        # 2. Test capture basique
        capture_ok = test_audio_capture_basic()
        
        # 3. Affichage résultats
        print("\n" + "="*70)
        print("🎤 RÉSULTATS VALIDATION AUDIO DEVICES PIPELINE")
        print("="*70)
        
        print(f"✅ Plateforme: {results['platform']}")
        print(f"✅ APIs Audio: {', '.join(results['available_apis'])}")
        
        print(f"\n📥 DEVICES D'ENTRÉE ({len(results['input_devices'])}):")
        for device in results['input_devices']:
            print(f"   • {device['name']} ({device['channels']}ch, {device['sample_rate']}Hz)")
        
        if results['default_input']:
            print(f"   → Défaut: {results['default_input']['name']}")
        
        print(f"\n📤 DEVICES DE SORTIE ({len(results['output_devices'])}):")
        for device in results['output_devices']:
            print(f"   • {device['name']} ({device['channels']}ch, {device['sample_rate']}Hz)")
        
        if results['default_output']:
            print(f"   → Défaut: {results['default_output']['name']}")
        
        print(f"\n🎯 TEST CAPTURE: {'✅ OK' if capture_ok else '❌ ÉCHEC'}")
        print(f"🔒 PERMISSIONS: {'✅ OK' if results['permissions_ok'] else '❌ ÉCHEC'}")
        
        print("="*70)
        if results['permissions_ok'] and capture_ok:
            print("🚀 PIPELINE AUTORISÉ - Configuration audio conforme")
        else:
            print("🛑 PIPELINE BLOQUÉ - Corriger configuration audio")
        print("="*70)
        
        return 0 if (results['permissions_ok'] and capture_ok) else 1
        
    except Exception as e:
        print("\n" + "="*70)
        print("🚫 ÉCHEC VALIDATION AUDIO DEVICES PIPELINE")
        print("="*70)
        print(f"❌ ERREUR: {e}")
        print("\n🔧 ACTIONS REQUISES:")
        print("   - Vérifier permissions microphone Windows")
        print("   - Installer PyAudio: pip install pyaudio")
        print("   - Configurer device audio par défaut")
        print("   - Tester microphone dans Paramètres Windows")
        print("="*70)
        print("🛑 PIPELINE BLOQUÉ - Corriger configuration audio avant continuation")
        print("="*70)
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 