#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test rapide de la voix française avec phonémisation IPA correcte
"""

import sys
import os
import time

# Configuration RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_french_voice():
    """Test de la voix française corrigée"""
    
    print("🇫🇷 TEST VOIX FRANÇAISE CORRIGÉE")
    print("=" * 50)
    
    # Configuration
    config = {
        'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
        'config_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json',
        'use_gpu': True
    }
    
    try:
        # Import du handler français
        print("1. 🚀 Import handler français...")
        from TTS.tts_handler_piper_french import TTSHandlerPiperFrench
        print("✅ Handler français importé")
        
        # Initialisation
        print("\n2. 🇫🇷 Initialisation...")
        start_time = time.time()
        handler = TTSHandlerPiperFrench(config)
        init_time = time.time() - start_time
        print(f"✅ Initialisé en {init_time:.2f}s")
        
        # Tests français progressifs
        test_phrases = [
            "Bonjour, je suis LUXA !",
            "Comment allez-vous aujourd'hui ?",
            "Je parle français avec une voix naturelle et claire.",
            "Voici un test de prononciation française avec des mots spéciaux : été, être, château, français."
        ]
        
        print("\n3. 🎯 Tests voix française...")
        
        for i, phrase in enumerate(test_phrases):
            print(f"\n   Test {i+1}: \"{phrase}\"")
            
            start_time = time.time()
            audio_data = handler.synthesize(phrase)
            synth_time = time.time() - start_time
            
            if len(audio_data) > 0:
                amplitude = max(abs(audio_data.min()), abs(audio_data.max()))
                chars_per_sec = len(phrase) / synth_time
                
                print(f"   ⚡ Temps: {synth_time:.3f}s ({chars_per_sec:.0f} car/s)")
                print(f"   🎵 Échantillons: {len(audio_data)}")
                print(f"   🔊 Amplitude: {amplitude}")
                
                if amplitude > 1000:  # Seuil pour vrai audio
                    print(f"   ✅ VOIX FRANÇAISE VALIDE")
                    
                    # Lecture du dernier test
                    if i == len(test_phrases) - 1:
                        print(f"   🔊 Écoute de la voix française...")
                        handler.speak(phrase)
                else:
                    print(f"   ⚠️ Audio faible")
            else:
                print(f"   ❌ Échec")
                
        print("\n🎉 Test français terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🇫🇷 LUXA - TEST VOIX FRANÇAISE")
    print("🎯 Objectif: Vérifier la phonémisation française IPA")
    print()
    
    success = test_french_voice()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TEST VOIX FRANÇAISE COMPLÉTÉ")
        print("💡 La voix devrait maintenant sonner français !")
    else:
        print("🚨 PROBLÈME VOIX FRANÇAISE")
    print("=" * 50) 