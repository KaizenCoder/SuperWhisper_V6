#!/usr/bin/env python3
"""
TEST SON SIMPLE LUXA - Juste faire parler l'assistant
🚨 RTX 3090 (CUDA:1) - SON AUDIBLE GARANTI
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090

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