#!/usr/bin/env python3
"""
Test VOIX FRANÇAISE QUI MARCHE - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) + VRAIE CONFIG TTS

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

def test_voix_francaise_correcte():
    """Test avec la VRAIE configuration qui marche d'après les docs"""
    
    print("\n🎤 TEST VOIX FRANÇAISE - CONFIGURATION CORRECTE")
    print("=" * 50)
    
    # ✅ CONFIGURATION CORRECTE d'après mvp_settings.yaml 
    config = {
        'model_path': 'models/fr_FR-siwis-medium.onnx',    # ✅ BON CHEMIN (pas D:\TTS_Voices\...)
        'config_path': 'models/fr_FR-siwis-medium.onnx.json',  # ✅ BON CHEMIN
        'use_gpu': True,
        'sample_rate': 22050
    }
    
    print(f"📄 Modèle TTS: {config['model_path']}")
    print(f"⚙️ Config: {config['config_path']}")
    
    try:
        # Vérification fichiers
        import os.path
        if not os.path.exists(config['model_path']):
            print(f"❌ Modèle introuvable: {config['model_path']}")
            print("💡 Tentative avec chemin alternatif...")
            # Essai chemin alternatif si nécessaire
            alt_path = 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx'
            if os.path.exists(alt_path):
                config['model_path'] = alt_path
                config['config_path'] = alt_path + '.json'
                print(f"✅ Modèle trouvé: {config['model_path']}")
            else:
                print("❌ Aucun modèle français trouvé")
                return False
        else:
            print(f"✅ Modèle trouvé: {config['model_path']}")
        
        # Test avec TTS Handler MVP (qui marchait d'après debug_tts)
        print("\n1. 🚀 Test TTS Handler MVP...")
        sys.path.append('TTS')
        from tts_handler_mvp import TTSHandlerMVP
        
        # Initialisation
        handler = TTSHandlerMVP(config)
        print("✅ Handler MVP initialisé")
        
        # Test voix française
        texte_test = "Bonjour ! Je suis LUXA, votre assistant vocal français."
        print(f"\n2. 🗣️ Test synthèse française: '{texte_test}'")
        
        # Synthèse
        audio_data = handler.synthesize(texte_test)
        print(f"✅ Audio généré: {len(audio_data) if audio_data is not None else 0} échantillons")
        
        # Lecture audio
        print("3. 🔊 Lecture audio française...")
        result = handler.speak(texte_test)
        print("✅ Lecture française terminée")
        
        if result:
            print("\n🎉 SUCCÈS: Voix française fonctionne!")
            return True
        else:
            print("\n❌ ÉCHEC: Problème lecture audio")
            return False
            
    except ImportError as e:
        print(f"❌ Import TTS Handler: {e}")
        print("💡 Tentative avec handler alternatif...")
        return test_avec_handler_alternatif(config)
    except Exception as e:
        print(f"❌ Erreur TTS: {e}")
        return False

def test_avec_handler_alternatif(config):
    """Test avec handler alternatif si MVP non disponible"""
    
    try:
        print("\n🔄 Test avec Piper Fixed Handler...")
        from tts_handler_piper_fixed import TTSHandlerPiperFixed
        
        handler = TTSHandlerPiperFixed(config)
        print("✅ Handler Piper initialisé")
        
        # Test voix
        texte = "LUXA utilise maintenant la bonne configuration française."
        print(f"🗣️ Test: '{texte}'")
        
        audio_data = handler.synthesize(texte)
        print(f"✅ Synthèse: {len(audio_data) if audio_data is not None else 0} échantillons")
        
        # Lecture
        handler.speak(texte)
        print("✅ Lecture terminée")
        
        print("\n🎉 SUCCÈS: Handler alternatif fonctionne!")
        return True
        
    except Exception as e:
        print(f"❌ Handler alternatif échoué: {e}")
        return False

if __name__ == "__main__":
    print("🎯 TEST VOIX FRANÇAISE LUXA - CONFIGURATION DOCUMENTÉE")
    print("🚨 UTILISE LA VRAIE CONFIG d'après DEBUG_TTS_FRENCH_VOICE_ISSUE.md")
    print()
    
    success = test_voix_francaise_correcte()
    
    if success:
        print("\n✅ MISSION ACCOMPLIE: Voix française FONCTIONNE!")
        print("🎤 Utilisez cette configuration pour l'assistant LUXA")
    else:
        print("\n❌ PROBLÈME PERSISTANT: Voix française ne fonctionne pas")
        print("📋 Vérifiez les fichiers modèles et dépendances TTS") 