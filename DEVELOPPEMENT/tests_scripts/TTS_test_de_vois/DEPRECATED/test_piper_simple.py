#!/usr/bin/env python3
"""
Test simple du handler TTS Piper

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test des importations nécessaires"""
    try:
        import piper
        print("✅ piper importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur import piper: {e}")
        return False
    
    try:
        import sounddevice
        print("✅ sounddevice importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur import sounddevice: {e}")
        return False
    
    try:
        import soundfile
        print("✅ soundfile importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur import soundfile: {e}")
        return False
    
    return True

def test_handler_import():
    """Test de l'importation du handler"""
    try:
        from TTS.tts_handler_piper import TTSHandlerPiper
        print("✅ TTSHandlerPiper importé avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur import TTSHandlerPiper: {e}")
        return False

def main():
    print("🧪 Test du système TTS Piper")
    print("=" * 40)
    
    # Test des importations de base
    print("\n1. Test des modules de base:")
    if not test_imports():
        print("❌ Échec des importations de base")
        return
    
    # Test de l'importation du handler
    print("\n2. Test du handler TTS:")
    if not test_handler_import():
        print("❌ Échec de l'importation du handler")
        return
    
    print("\n🎉 Tous les tests sont passés avec succès !")
    print("Le système TTS Piper est prêt à être utilisé.")

if __name__ == "__main__":
    main() 