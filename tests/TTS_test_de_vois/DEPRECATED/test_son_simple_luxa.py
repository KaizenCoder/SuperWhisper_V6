#!/usr/bin/env python3
"""
TEST SON SIMPLE LUXA - Juste faire parler l'assistant
🚨 RTX 3090 (CUDA:1) - SON AUDIBLE GARANTI

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

def test_son_simple():
    """Test ultra-simple pour entendre la voix"""
    
    print("🎤 TEST SON SIMPLE LUXA")
    print("=" * 30)
    
    try:
        # Import simple
        import sys
        sys.path.append('TTS')
        from tts_handler_mvp import TTSHandlerMVP
        
        # Config minimale
        config = {'use_gpu': True}
        
        # Initialisation
        print("1. 🚀 Initialisation...")
        handler = TTSHandlerMVP(config)
        print("✅ Handler OK")
        
        # Texte simple
        texte = "Bonjour, je suis LUXA."
        print(f"2. 🗣️ Texte: '{texte}'")
        
        # JUSTE FAIRE PARLER
        print("3. 🔊 Lecture...")
        handler.speak(texte)
        print("✅ Terminé!")
        
        print("\n🎉 Si vous avez entendu la voix, ça marche!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    test_son_simple() 