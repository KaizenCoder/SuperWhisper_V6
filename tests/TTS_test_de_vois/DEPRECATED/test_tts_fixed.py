#!/usr/bin/env python3
"""
Test du handler TTS Piper corrigé avec phonémisation correcte

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

# Configuration RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tts_fixed():
    """Test du handler TTS corrigé"""
    
    print("🔧 TEST TTS PIPER CORRIGÉ")
    print("=" * 50)
    
    # Configuration
    config = {
        'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
        'config_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json',
        'use_gpu': True
    }
    
    try:
        # Import du handler corrigé
        print("1. 📦 Import handler corrigé...")
        from TTS.tts_handler_piper_fixed import TTSHandlerPiperFixed
        print("✅ Handler corrigé importé")
        
        # Initialisation
        print("\n2. 🚀 Initialisation...")
        start_time = time.time()
        handler = TTSHandlerPiperFixed(config)
        init_time = time.time() - start_time
        print(f"✅ Initialisé en {init_time:.2f}s")
        
        # Tests progressifs
        test_texts = [
            "bonjour",         # Test simple
            "salut !",         # Avec ponctuation
            "comment allez-vous ?",  # Plus long
        ]
        
        print("\n3. 🎯 Tests de synthèse...")
        
        for i, text in enumerate(test_texts):
            print(f"\n   Test {i+1}: \"{text}\"")
            
            start_time = time.time()
            audio_data = handler.synthesize(text)
            synth_time = time.time() - start_time
            
            if len(audio_data) > 0:
                # Vérifier si l'audio contient vraiment du son
                audio_range = [audio_data.min(), audio_data.max()]
                audio_amplitude = max(abs(audio_data.min()), abs(audio_data.max()))
                
                print(f"   ⚡ Temps: {synth_time:.3f}s")
                print(f"   🎵 Échantillons: {len(audio_data)}")
                print(f"   🔊 Amplitude: {audio_amplitude}")
                
                if audio_amplitude > 100:  # Seuil minimum pour du vrai son
                    print(f"   ✅ AUDIO VALIDE - Son détecté !")
                    
                    # Lecture du dernier test seulement
                    if i == len(test_texts) - 1:
                        print(f"   🔊 Lecture audio test...")
                        handler.speak(text)
                else:
                    print(f"   ⚠️ Audio faible ou silencieux")
            else:
                print(f"   ❌ Échec synthèse")
                
        print("\n🎉 Test terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tts_fixed()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 HANDLER TTS CORRIGÉ TESTÉ")
    else:
        print("🚨 PROBLÈME AVEC HANDLER CORRIGÉ")
    print("=" * 50) 