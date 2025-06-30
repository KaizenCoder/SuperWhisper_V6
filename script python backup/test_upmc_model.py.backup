#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du nouveau modèle Piper français fr_FR-upmc-medium
"""

import sys
import os
import time

# Configuration RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_upmc_model():
    """Test du nouveau modèle fr_FR-upmc-medium"""
    
    print("🇫🇷 TEST NOUVEAU MODÈLE PIPER FRANÇAIS UPMC")
    print("=" * 60)
    print("🎯 Objectif: Tester le modèle fr_FR-upmc-medium recommandé par ChatGPT")
    print("📋 Comparaison avec notre solution SAPI temporaire")
    print()
    
    # Configuration pour le nouveau modèle
    config = {
        'model_path': 'models/fr_FR-upmc-medium.onnx',
        'config_path': 'models/fr_FR-upmc-medium.onnx.json',
        'use_gpu': True,
        'sample_rate': 22050
    }
    
    try:
        # Import du handler Piper corrigé
        print("1. 🚀 Import handler Piper français corrigé...")
        from TTS.tts_handler_piper_french import TTSHandlerPiperFrench
        print("✅ Handler Piper français importé")
        
        # Initialisation avec nouveau modèle
        print("\n2. 🔧 Initialisation nouveau modèle UPMC...")
        start_time = time.time()
        handler = TTSHandlerPiperFrench(config)
        init_time = time.time() - start_time
        print(f"✅ Modèle UPMC initialisé en {init_time:.2f}s")
        
        # Tests comparatifs français
        print("\n3. 🎯 Tests comparatifs français...")
        test_phrases = [
            "Bonjour, je suis LUXA avec le modèle UPMC.",
            "Ce modèle devrait avoir une meilleure prononciation française.",
            "Comparons avec Microsoft Hortense."
        ]
        
        all_success = True
        total_chars = 0
        total_time = 0
        
        for i, phrase in enumerate(test_phrases):
            print(f"\n   Test UPMC {i+1}: \"{phrase}\"")
            
            start_time = time.time()
            audio_data = handler.synthesize(phrase)
            synth_time = time.time() - start_time
            
            if len(audio_data) > 0:
                chars_per_sec = len(phrase) / synth_time if synth_time > 0 else 0
                amplitude = max(abs(audio_data.min()), abs(audio_data.max()))
                
                print(f"      ⚡ Temps UPMC: {synth_time:.3f}s ({chars_per_sec:.0f} car/s)")
                print(f"      🎵 Données UPMC: {len(audio_data)} échantillons")
                print(f"      🔊 Amplitude UPMC: {amplitude}")
                
                if amplitude > 1000:
                    print(f"      ✅ Synthèse UPMC réussie")
                    total_chars += len(phrase)
                    total_time += synth_time
                else:
                    print(f"      ⚠️ Audio UPMC faible")
                    all_success = False
            else:
                print(f"      ❌ Échec synthèse UPMC")
                all_success = False
        
        # Test audio final avec lecture
        print("\n4. 🔊 Test lecture audio UPMC final...")
        test_audio_phrase = "Test final du modèle UPMC français. Cette voix sonne-t-elle vraiment française maintenant ?"
        print(f"   Phrase critique: \"{test_audio_phrase}\"")
        
        start_time = time.time()
        audio_data = handler.speak(test_audio_phrase)
        speak_time = time.time() - start_time
        
        if len(audio_data) > 0:
            print(f"   ✅ Lecture UPMC réussie en {speak_time:.2f}s")
            print(f"   ❓ QUESTION CRITIQUE: Est-ce que cette voix sonne française ?")
        else:
            print(f"   ❌ Échec lecture UPMC")
            all_success = False
        
        # Résultats finaux
        print("\n5. 📊 Résultats UPMC vs SAPI:")
        if total_time > 0:
            avg_speed = total_chars / total_time
            print(f"   Performance UPMC: {avg_speed:.0f} caractères/seconde")
            print(f"   Performance SAPI: ~1,134 caractères/seconde (référence)")
        
        if all_success:
            print("   ✅ Tous les tests UPMC réussis")
            print("   🎉 Prêt pour comparaison qualité française")
        else:
            print("   ⚠️ Certains tests UPMC échoués")
        
        return all_success
        
    except Exception as e:
        print(f"❌ Erreur test UPMC: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🇫🇷 LUXA - TEST MODÈLE PIPER UPMC")
    print("🎯 Validation du nouveau modèle français recommandé")
    print("📋 Comparaison avec solution SAPI temporaire")
    print()
    
    success = test_upmc_model()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MODÈLE UPMC TESTÉ !")
        print("❓ QUESTION: La voix UPMC sonne-t-elle plus française que SAPI ?")
        print("💡 Si OUI → Remplacer SAPI par UPMC dans MVP")
        print("💡 Si NON → Garder SAPI pour MVP P0")
    else:
        print("🚨 PROBLÈME MODÈLE UPMC")
        print("✅ Solution: Garder SAPI pour MVP P0")
    print("=" * 60) 