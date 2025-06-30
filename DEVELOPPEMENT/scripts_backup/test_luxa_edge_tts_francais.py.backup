#!/usr/bin/env python3
"""
TEST LUXA_TTS - Microsoft Edge TTS FRANÇAIS
🚨 RTX 3090 (CUDA:1) - VOIX FRANÇAISE PREMIUM MICROSOFT
"""

import os
import sys

# Configuration RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU RTX 3090 configurée")
print("🇫🇷 TEST LUXA_TTS - MICROSOFT EDGE TTS FRANÇAIS")

def test_luxa_edge_tts():
    """Test LUXA_TTS avec Microsoft Edge TTS français"""
    
    print("\n🎭 TEST LUXA_TTS - EDGE TTS FRANÇAIS")
    print("=" * 50)
    
    try:
        # Import du vrai système LUXA_TTS
        sys.path.append('LUXA_TTS')
        from tts_handler import TTSHandler
        
        # Configuration pour Edge TTS français
        config = {
            'voice': 'fr-FR-DeniseNeural',  # Voix française premium
            'rate': '+0%',
            'volume': '+0%'
        }
        
        print("🚀 Initialisation LUXA_TTS Edge TTS...")
        handler = TTSHandler(config)
        print("✅ Handler LUXA_TTS initialisé")
        
        # Texte français complet
        texte = "Bonjour ! Je suis LUXA, votre assistant vocal intelligent. J'utilise Microsoft Edge TTS avec une voix française premium pour vous offrir une expérience naturelle et agréable."
        print(f"📝 Texte: {texte[:50]}...")
        
        # Test synthèse
        print("🎤 Synthèse avec fr-FR-DeniseNeural...")
        handler.speak(texte)
        print("✅ Synthèse terminée")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur LUXA_TTS Edge: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_autres_voix_francaises():
    """Test d'autres voix françaises Microsoft Edge disponibles"""
    
    print("\n🎭 TEST AUTRES VOIX FRANÇAISES MICROSOFT")
    print("=" * 50)
    
    voix_francaises = [
        'fr-FR-DeniseNeural',      # Voix féminine premium
        'fr-FR-HenriNeural',       # Voix masculine premium  
        'fr-FR-AlainNeural',       # Voix masculine alternative
        'fr-FR-BrigitteNeural',    # Voix féminine alternative
        'fr-FR-CelesteNeural',     # Voix féminine jeune
        'fr-FR-ClaudeNeural',      # Voix masculine mature
    ]
    
    for voix in voix_francaises:
        print(f"\n🎤 Test voix: {voix}")
        
        try:
            sys.path.append('LUXA_TTS')
            from tts_handler import TTSHandler
            
            # Configuration pour cette voix
            config = {
                'voice': voix,
                'rate': '+0%',
                'volume': '+0%'
            }
            
            handler = TTSHandler(config)
            handler.voice = voix  # Override la voix
            
            # Texte court pour test
            texte = f"Bonjour, je suis {voix.split('-')[2].replace('Neural', '')}."
            print(f"📝 Test: {texte}")
            
            handler.speak(texte)
            print(f"✅ {voix} : Succès")
            
        except Exception as e:
            print(f"❌ {voix} : Erreur - {e}")

if __name__ == "__main__":
    print("🎯 DÉMARRAGE TEST LUXA_TTS")
    print("=" * 60)
    
    # Test principal avec voix premium
    succes = test_luxa_edge_tts()
    
    if succes:
        print("\n🎉 SUCCÈS PRINCIPAL ! Testez d'autres voix...")
        
        # Demander si on teste d'autres voix
        print("\n🎤 Voulez-vous tester d'autres voix françaises ? (y/N)")
        response = input("Réponse: ").strip().lower()
        
        if response in ['y', 'yes', 'oui', 'o']:
            test_autres_voix_francaises()
    else:
        print("\n❌ ÉCHEC : Problème avec LUXA_TTS Edge")
    
    print("\n🏁 FIN DES TESTS LUXA_TTS") 