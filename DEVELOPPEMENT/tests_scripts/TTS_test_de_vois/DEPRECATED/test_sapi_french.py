#!/usr/bin/env python3
"""
Test voix française avec SAPI Windows natif

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

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_sapi_french():
    """Test voix française avec SAPI Windows"""
    
    print("🇫🇷 TEST VOIX FRANÇAISE SAPI WINDOWS")
    print("=" * 50)
    
    # Configuration
    config = {
        'sample_rate': 22050
    }
    
    try:
        # Import du handler SAPI
        print("1. 🚀 Import handler SAPI français...")
        from TTS.tts_handler_sapi_french import TTSHandlerSapiFrench
        print("✅ Handler SAPI français importé")
        
        # Initialisation
        print("\n2. 🇫🇷 Initialisation SAPI...")
        start_time = time.time()
        handler = TTSHandlerSapiFrench(config)
        init_time = time.time() - start_time
        print(f"✅ Initialisé en {init_time:.2f}s")
        
        # Tests français avec SAPI
        test_phrases = [
            "Bonjour, je suis LUXA !",
            "Voici un test de voix française Windows.",
            "Cette voix utilise le système SAPI natif.",
        ]
        
        print("\n3. 🎯 Tests voix SAPI française...")
        
        for i, phrase in enumerate(test_phrases):
            print(f"\n   Test {i+1}: \"{phrase}\"")
            
            start_time = time.time()
            audio_data = handler.synthesize(phrase)
            synth_time = time.time() - start_time
            
            if len(audio_data) > 0:
                amplitude = max(abs(audio_data.min()), abs(audio_data.max()))
                chars_per_sec = len(phrase) / synth_time if synth_time > 0 else 0
                
                print(f"   ⚡ Temps: {synth_time:.3f}s ({chars_per_sec:.0f} car/s)")
                print(f"   🎵 Échantillons: {len(audio_data)}")
                print(f"   🔊 Amplitude: {amplitude}")
                
                if amplitude > 1000:  # Seuil pour vrai audio
                    print(f"   ✅ VOIX SAPI FRANÇAISE VALIDE")
                    
                    # Lecture du dernier test
                    if i == len(test_phrases) - 1:
                        print(f"   🔊 Écoute de la voix SAPI française...")
                        handler.speak(phrase)
                else:
                    print(f"   ⚠️ Audio faible")
            else:
                print(f"   ❌ Échec")
        
        print("\n🎉 Test SAPI français terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🇫🇷 LUXA - TEST VOIX SAPI FRANÇAISE")
    print("🎯 Objectif: Tester voix française Windows native")
    print()
    
    success = test_sapi_french()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TEST SAPI FRANÇAIS COMPLÉTÉ")
        print("💡 La voix Windows française native !")
    else:
        print("🚨 PROBLÈME SAPI FRANÇAIS")
    print("=" * 50) 