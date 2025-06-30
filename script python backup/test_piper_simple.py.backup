#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple du handler TTS Piper
"""

import sys
import os

# Ajouter le répertoire courant au path
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