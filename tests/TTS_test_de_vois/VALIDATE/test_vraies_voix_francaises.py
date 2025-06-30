#!/usr/bin/env python3
"""
TEST VRAIES VOIX FRANÇAISES - Solutions alternatives
🚨 RTX 3090 (CUDA:1) - RECHERCHE VOIX FRANÇAISE QUI MARCHE VRAIMENT

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

def test_windows_sapi_francais():
    """Test voix SAPI Windows en français"""
    
    print("\n🎤 TEST SAPI WINDOWS FRANÇAIS")
    print("=" * 40)
    
    try:
        import win32com.client
        
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voices = speaker.GetVoices()
        
        print("🔍 Voix disponibles:")
        voix_francaises = []
        
        for i in range(voices.Count):
            voice = voices.Item(i)
            voice_name = voice.GetDescription()
            print(f"   {i}: {voice_name}")
            
            # Chercher voix françaises
            if any(mot in voice_name.lower() for mot in ["french", "français", "france", "hortense", "julie"]):
                voix_francaises.append((i, voice, voice_name))
        
        if voix_francaises:
            print(f"\n🇫🇷 {len(voix_francaises)} voix françaises trouvées:")
            
            for idx, (i, voice, name) in enumerate(voix_francaises):
                print(f"\n--- Test voix française {idx+1}: {name} ---")
                speaker.Voice = voice
                
                texte = f"Bonjour ! Je suis LUXA. Ceci est un test avec la voix {name.split()[-1] if name else 'française'}. Est-ce que vous m'entendez bien parler français ?"
                
                print(f"🗣️ Texte: {texte[:80]}...")
                print("🔊 Lecture SAPI français...")
                speaker.Speak(texte)
                print("✅ Test terminé")
                
                # Demander feedback utilisateur
                print("❓ Cette voix parle-t-elle français ? (Entrée pour continuer)")
                input()
            
            return True
        else:
            print("❌ Aucune voix française SAPI trouvée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur SAPI: {e}")
        return False

def test_edge_tts_francais():
    """Test Microsoft Edge TTS français (si disponible)"""
    
    print("\n🌐 TEST MICROSOFT EDGE TTS FRANÇAIS")
    print("=" * 45)
    
    try:
        import edge_tts
        import asyncio
        import tempfile
        import soundfile as sf
        import sounddevice as sd
        
        async def generate_french_speech():
            # Voix françaises Edge TTS
            voix_francaises = [
                "fr-FR-DeniseNeural",
                "fr-FR-HenriNeural", 
                "fr-CA-AntoineNeural",
                "fr-CA-JeanNeural",
                "fr-CH-FabriceNeural"
            ]
            
            texte = "Bonjour ! Je suis LUXA, votre assistant vocal. Cette voix provient de Microsoft Edge TTS français."
            
            for voix in voix_francaises:
                print(f"\n🎤 Test voix: {voix}")
                
                try:
                    communicate = edge_tts.Communicate(texte, voix)
                    
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                        await communicate.save(tmp.name)
                        
                        # Conversion et lecture
                        audio, sr = sf.read(tmp.name)
                        
                        print(f"🔊 Lecture {voix}...")
                        sd.play(audio, samplerate=sr)
                        sd.wait()
                        print("✅ Lecture terminée")
                        
                        os.unlink(tmp.name)
                        
                        print("❓ Cette voix parle-t-elle français ? (Entrée pour continuer)")
                        input()
                        
                except Exception as e:
                    print(f"❌ Erreur voix {voix}: {e}")
        
        asyncio.run(generate_french_speech())
        return True
        
    except ImportError:
        print("❌ edge-tts non installé. Installation: pip install edge-tts")
        return False
    except Exception as e:
        print(f"❌ Erreur Edge TTS: {e}")
        return False

def test_gtts_francais():
    """Test Google TTS français (si disponible)"""
    
    print("\n🌍 TEST GOOGLE TTS FRANÇAIS")
    print("=" * 35)
    
    try:
        from gtts import gTTS
        import tempfile
        import soundfile as sf
        import sounddevice as sd
        
        texte = "Bonjour ! Je suis LUXA avec Google TTS français. Cette voix devrait être parfaitement française."
        
        print("🗣️ Génération Google TTS français...")
        tts = gTTS(text=texte, lang='fr', slow=False)
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tts.save(tmp.name)
            
            # Lecture
            audio, sr = sf.read(tmp.name)
            
            print("🔊 Lecture Google TTS français...")
            sd.play(audio, samplerate=sr)
            sd.wait()
            print("✅ Lecture terminée")
            
            os.unlink(tmp.name)
            
        return True
        
    except ImportError:
        print("❌ gTTS non installé. Installation: pip install gtts")
        return False
    except Exception as e:
        print(f"❌ Erreur gTTS: {e}")
        return False

def test_festival_francais():
    """Test Festival TTS français local (si installé)"""
    
    print("\n🎭 TEST FESTIVAL TTS FRANÇAIS")
    print("=" * 35)
    
    try:
        import subprocess
        import tempfile
        import soundfile as sf
        import sounddevice as sd
        
        texte = "Bonjour ! Je suis LUXA avec Festival français."
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Commande Festival avec voix française
            cmd = [
                'festival', 
                '--language', 'french',
                '--tts',
                '--output', tmp.name
            ]
            
            process = subprocess.run(
                cmd,
                input=texte,
                text=True,
                capture_output=True
            )
            
            if process.returncode == 0 and os.path.exists(tmp.name):
                audio, sr = sf.read(tmp.name)
                
                print("🔊 Lecture Festival français...")
                sd.play(audio, samplerate=sr)
                sd.wait()
                print("✅ Lecture terminée")
                
                os.unlink(tmp.name)
                return True
            else:
                print("❌ Festival échoué ou non installé")
                return False
                
    except FileNotFoundError:
        print("❌ Festival non installé")
        return False
    except Exception as e:
        print(f"❌ Erreur Festival: {e}")
        return False

if __name__ == "__main__":
    print("🎯 RECHERCHE VRAIE VOIX FRANÇAISE POUR LUXA")
    print("=" * 60)
    
    solutions = [
        ("SAPI Windows", test_windows_sapi_francais),
        ("Microsoft Edge TTS", test_edge_tts_francais),
        ("Google TTS", test_gtts_francais),
        ("Festival TTS", test_festival_francais)
    ]
    
    for nom, test_func in solutions:
        print(f"\n🔄 Test solution: {nom}")
        try:
            success = test_func()
            if success:
                print(f"✅ {nom} : Test réussi")
            else:
                print(f"⚠️ {nom} : Test échoué")
        except Exception as e:
            print(f"❌ {nom} : Erreur {e}")
    
    print("\n🎯 Tests terminés. Quelle solution a produit la meilleure voix française ?") 