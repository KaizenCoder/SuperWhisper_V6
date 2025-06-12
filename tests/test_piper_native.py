#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du modèle français avec Piper CLI natif
"""

import sys
import os
import time

# Configuration RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_piper_native():
    """Test du modèle français avec CLI natif Piper"""
    
    print("🇫🇷 TEST MODÈLE FRANÇAIS PIPER NATIF")
    print("=" * 50)
    print("🎯 Objectif: Vérifier si le modèle fr_FR-siwis-medium produit VRAIMENT du français")
    print()
    
    # Configuration
    config = {
        'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
        'sample_rate': 22050
    }
    
    try:
        # Import du handler natif
        print("1. 🚀 Import handler Piper natif...")
        from TTS.tts_handler_piper_native import TTSHandlerPiperNative
        print("✅ Handler Piper natif importé")
        
        # Initialisation
        print("\n2. 🔧 Initialisation Piper natif...")
        start_time = time.time()
        handler = TTSHandlerPiperNative(config)
        init_time = time.time() - start_time
        print(f"✅ Initialisé en {init_time:.2f}s")
        
        # Tests français critiques
        test_phrases = [
            "Bonjour",
            "Je suis français",
            "Voici un test de prononciation française"
        ]
        
        print("\n3. 🎯 Tests modèle ORIGINAL...")
        print("💡 ÉCOUTER ATTENTIVEMENT : Est-ce que ça sonne français ?")
        
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
                    print(f"   ✅ AUDIO NATIF VALIDE")
                    
                    # Lecture du dernier test
                    if i == len(test_phrases) - 1:
                        print(f"   🔊 🇫🇷 ÉCOUTE CRITIQUE - MODÈLE NATIF...")
                        handler.speak(phrase)
                        print(f"   ❓ Question: Est-ce que cette voix sonne française ?")
                else:
                    print(f"   ⚠️ Audio faible")
            else:
                print(f"   ❌ Échec synthèse")
        
        print("\n🎉 Test modèle natif terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🇫🇷 LUXA - TEST MODÈLE FRANÇAIS NATIF")
    print("🎯 Objectif: Diagnostiquer si le problème vient du modèle ou de notre code")
    print()
    
    success = test_piper_native()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST MODÈLE NATIF COMPLÉTÉ")
        print("💡 Résultat critique:")
        print("   - Si la voix sonne française → Notre phonémisation était mauvaise")
        print("   - Si la voix ne sonne PAS française → Le modèle n'est pas français")
    else:
        print("🚨 PROBLÈME TECHNIQUE")
    print("=" * 60) 