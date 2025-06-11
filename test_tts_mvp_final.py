#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test final TTS MVP avec voix française Windows (Microsoft Hortense)
"""

import sys
import os
import time

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tts_mvp_final():
    """Test final TTS MVP français"""
    
    print("🇫🇷 TEST FINAL TTS MVP FRANÇAIS")
    print("=" * 60)
    print("🎯 Objectif: Valider le handler MVP avec Microsoft Hortense")
    print("📋 Contexte: Voix française Windows native pour LUXA MVP P0")
    print()
    
    # Configuration MVP
    config = {
        'sample_rate': 22050,
        'use_gpu': False
    }
    
    try:
        # Import du handler MVP
        print("1. 🚀 Import handler TTS MVP...")
        from TTS.tts_handler_mvp import TTSHandlerMVP
        print("✅ Handler TTS MVP importé")
        
        # Initialisation
        print("\n2. 🔧 Initialisation TTS MVP...")
        start_time = time.time()
        handler = TTSHandlerMVP(config)
        init_time = time.time() - start_time
        print(f"✅ TTS MVP initialisé en {init_time:.2f}s")
        
        # Informations handler
        print("\n3. ℹ️ Informations TTS MVP:")
        info = handler.get_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Tests de synthèse MVP
        print("\n4. 🎯 Tests synthèse MVP française...")
        test_phrases = [
            "Bonjour, je suis LUXA.",
            "Votre assistant personnel français.",
            "Prêt pour la phase de développement MVP."
        ]
        
        all_success = True
        total_chars = 0
        total_time = 0
        
        for i, phrase in enumerate(test_phrases):
            print(f"\n   Test MVP {i+1}: \"{phrase}\"")
            
            start_time = time.time()
            audio_data = handler.synthesize(phrase)
            synth_time = time.time() - start_time
            
            if len(audio_data) > 0:
                chars_per_sec = len(phrase) / synth_time if synth_time > 0 else 0
                amplitude = max(abs(audio_data.min()), abs(audio_data.max()))
                
                print(f"      ⚡ Temps: {synth_time:.3f}s ({chars_per_sec:.0f} car/s)")
                print(f"      🎵 Données: {len(audio_data)} échantillons")
                print(f"      🔊 Amplitude: {amplitude}")
                
                if amplitude > 1000:
                    print(f"      ✅ Synthèse MVP réussie")
                    total_chars += len(phrase)
                    total_time += synth_time
                else:
                    print(f"      ⚠️ Audio MVP faible")
                    all_success = False
            else:
                print(f"      ❌ Échec synthèse MVP")
                all_success = False
        
        # Test avec lecture audio
        print("\n5. 🔊 Test lecture audio MVP...")
        test_audio_phrase = "LUXA MVP, test de lecture audio française."
        print(f"   Phrase: \"{test_audio_phrase}\"")
        
        start_time = time.time()
        audio_data = handler.speak(test_audio_phrase)
        speak_time = time.time() - start_time
        
        if len(audio_data) > 0:
            print(f"   ✅ Lecture MVP réussie en {speak_time:.2f}s")
        else:
            print(f"   ❌ Échec lecture MVP")
            all_success = False
        
        # Résultats finaux
        print("\n6. 📊 Résultats finaux MVP:")
        if total_time > 0:
            avg_speed = total_chars / total_time
            print(f"   Performance moyenne: {avg_speed:.0f} caractères/seconde")
        
        if all_success:
            print("   ✅ Tous les tests MVP réussis")
            print("   🎉 Handler TTS MVP prêt pour intégration")
        else:
            print("   ⚠️ Certains tests MVP échoués")
        
        return all_success
        
    except Exception as e:
        print(f"❌ Erreur test MVP: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🇫🇷 LUXA MVP - TEST TTS FRANÇAIS FINAL")
    print("🎯 Validation du handler MVP avec Microsoft Hortense")
    print("📋 Objectif: Confirmer la fonctionnalité pour MVP P0")
    print()
    
    success = test_tts_mvp_final()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TTS MVP FRANÇAIS VALIDÉ !")
        print("✅ Prêt pour intégration dans run_assistant.py")
        print("📋 Prochaine étape: Intégrer dans le pipeline MVP P0")
    else:
        print("🚨 PROBLÈME TTS MVP")
        print("⚠️ Correction nécessaire avant intégration")
    print("=" * 60) 