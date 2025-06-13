#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test voix française Windows SAPI directe
"""

import sys
import os

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