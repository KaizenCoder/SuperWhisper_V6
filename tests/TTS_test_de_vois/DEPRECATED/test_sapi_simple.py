#!/usr/bin/env python3
"""
Test voix française Windows SAPI directe

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

def test_sapi_simple():
    """Test voix française Windows SAPI"""
    
    print("🇫🇷 TEST VOIX FRANÇAISE WINDOWS SAPI")
    print("=" * 50)
    
    try:
        # Import win32com si disponible
        import win32com.client
        print("✅ win32com disponible")
        
        # Initialiser SAPI
        print("\\n1. 🔧 Initialisation SAPI...")
        sapi = win32com.client.Dispatch("SAPI.SpVoice")
        
        # Lister les voix disponibles
        voices = sapi.GetVoices()
        print(f"\\n2. 🔍 {voices.Count} voix Windows détectées:")
        
        french_voices = []
        for i in range(voices.Count):
            voice = voices.Item(i)
            name = voice.GetDescription()
            print(f"   Voix {i}: {name}")
            
            # Chercher voix française
            if any(keyword in name.lower() for keyword in ['french', 'français', 'france']):
                french_voices.append((i, voice, name))
                print(f"      ✅ VOIX FRANÇAISE DÉTECTÉE!")
        
        if french_voices:
            print(f"\\n3. 🇫🇷 Test avec voix française...")
            
            # Utiliser la première voix française
            voice_index, french_voice, voice_name = french_voices[0]
            sapi.Voice = french_voice
            print(f"   Voix sélectionnée: {voice_name}")
            
            # Test de synthèse
            test_text = "Bonjour, je suis LUXA, votre assistant vocal français."
            print(f"   Texte: '{test_text}'")
            
            print("   🔊 Synthèse en cours...")
            sapi.Speak(test_text)
            print("   ✅ Synthèse terminée")
            
            return True
        else:
            print("\\n⚠️ Aucune voix française détectée")
            print("💡 Test avec voix par défaut...")
            
            test_text = "Hello, this is a test with default voice."
            print(f"   Texte: '{test_text}'")
            
            print("   🔊 Synthèse en cours...")
            sapi.Speak(test_text)
            print("   ✅ Synthèse terminée")
            
            return False
            
    except ImportError:
        print("❌ win32com non disponible")
        print("💡 Installation: pip install pywin32")
        return False
    except Exception as e:
        print(f"❌ Erreur SAPI: {e}")
        return False

if __name__ == "__main__":
    print("🇫🇷 LUXA - TEST VOIX WINDOWS SIMPLE")
    print("🎯 Objectif: Tester voix française Windows native")
    print()
    
    has_french = test_sapi_simple()
    
    print("\\n" + "=" * 50)
    if has_french:
        print("🎉 VOIX FRANÇAISE WINDOWS TROUVÉE !")
        print("💡 Nous pouvons utiliser cette voix temporairement")
    else:
        print("⚠️ Pas de voix française Windows")
        print("💡 Fallback nécessaire")
    print("=" * 50) 