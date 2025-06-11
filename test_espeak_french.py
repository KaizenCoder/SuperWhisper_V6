#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test voix française avec espeak-ng authentique
"""

import sys
import os
import time

# Configuration RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_espeak_french():
    """Test voix française avec espeak-ng"""
    
    print("🇫🇷 TEST VOIX FRANÇAISE ESPEAK-NG")
    print("=" * 50)
    
    # Configuration
    config = {
        'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
        'config_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json',
        'use_gpu': True
    }
    
    try:
        # Import du handler espeak
        print("1. 🚀 Import handler espeak français...")
        from TTS.tts_handler_piper_espeak import TTSHandlerPiperEspeak
        print("✅ Handler espeak français importé")
        
        # Initialisation
        print("\n2. 🇫🇷 Initialisation espeak...")
        start_time = time.time()
        handler = TTSHandlerPiperEspeak(config)
        init_time = time.time() - start_time
        print(f"✅ Initialisé en {init_time:.2f}s")
        
        # Tests français avec espeak
        test_phrases = [
            "Bonjour, comment allez-vous ?",
            "Je suis LUXA, assistant vocal français.",
            "Voici un test de prononciation avec espeak-ng.",
        ]
        
        print("\n3. 🎯 Tests voix espeak française...")
        
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
                    print(f"   ✅ VOIX ESPEAK FRANÇAISE VALIDE")
                    
                    # Lecture du dernier test
                    if i == len(test_phrases) - 1:
                        print(f"   🔊 Écoute de la voix espeak française...")
                        handler.speak(phrase)
                else:
                    print(f"   ⚠️ Audio faible")
            else:
                print(f"   ❌ Échec")
        
        print("\n🎉 Test espeak français terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🇫🇷 LUXA - TEST VOIX ESPEAK FRANÇAISE")
    print("🎯 Objectif: Tester phonémisation espeak-ng authentique")
    print()
    
    success = test_espeak_french()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TEST ESPEAK FRANÇAIS COMPLÉTÉ")
        print("💡 La voix devrait maintenant sonner authentiquement française !")
    else:
        print("🚨 PROBLÈME ESPEAK FRANÇAIS")
    print("=" * 50) 