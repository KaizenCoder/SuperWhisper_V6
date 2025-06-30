#!/usr/bin/env python3
"""
Test voix naturelle LUXA - SuperWhisper V6
🎮 RTX 3090 (CUDA:1) - VOIX NATURELLE QUI MARCHE
"""

import os
import sys
import time

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def test_voix_windows_sapi_amelioree():
    """Test des voix Windows SAPI améliorées - VOIX NATURELLES"""
    print("\n🎤 TEST VOIX WINDOWS SAPI NATURELLES")
    print("=" * 60)
    print("🌟 Recherche voix naturelles installées sur Windows")
    print()
    
    try:
        import win32com.client
        
        # Créer objet SAPI
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voices = speaker.GetVoices()
        
        print(f"📊 {voices.Count} voix détectées sur le système")
        print()
        
        # Lister toutes les voix disponibles
        voix_naturelles = []
        for i in range(voices.Count):
            voice = voices.Item(i)
            name = voice.GetDescription()
            print(f"{i+1}. {name}")
            
            # Détecter voix naturelles (généralement celles avec "Neural", "Premium", ou certains noms)
            if any(keyword in name.lower() for keyword in ['neural', 'premium', 'natural', 'enhanced', 'clara', 'guillaume', 'julie', 'caroline']):
                voix_naturelles.append((i, name))
        
        print()
        if voix_naturelles:
            print(f"🌟 {len(voix_naturelles)} voix naturelles détectées:")
            for idx, name in voix_naturelles:
                print(f"   ✨ {name}")
        else:
            print("⚠️ Pas de voix naturelle premium détectée")
            print("💡 Utilisation de la meilleure voix disponible")
        
        print()
        
        # Test des voix naturelles
        test_message = "Bonjour ! Je suis LUXA, votre assistant vocal avec une voix naturelle améliorée."
        
        if voix_naturelles:
            print("🎯 Test des voix naturelles:")
            print()
            
            for i, (voice_idx, voice_name) in enumerate(voix_naturelles):
                print(f"   Test {i+1}: {voice_name}")
                speaker.Voice = voices.Item(voice_idx)
                
                # Réglages pour voix naturelle
                speaker.Rate = 0      # Vitesse normale
                speaker.Volume = 100  # Volume maximum
                
                print("   🔊 Écouter cette voix...")
                speaker.Speak(test_message)
                
                # Demander avis utilisateur
                while True:
                    choix = input("   💬 Cette voix vous plaît ? (o/n/q pour quitter): ").strip().lower()
                    if choix in ['o', 'oui', 'y', 'yes']:
                        print(f"   ✅ Voix sélectionnée: {voice_name}")
                        return voice_idx, voice_name, speaker
                    elif choix in ['n', 'non', 'no']:
                        print("   ➡️ Voix suivante...")
                        break
                    elif choix in ['q', 'quit', 'exit']:
                        return None, None, None
                    else:
                        print("   ❌ Réponse invalide (o/n/q)")
                
                print()
        
        # Si aucune voix naturelle sélectionnée, utiliser la meilleure disponible
        print("🔍 Sélection automatique de la meilleure voix...")
        
        # Chercher une voix française de qualité
        best_voice_idx = 0
        best_voice_name = "Voix par défaut"
        
        for i in range(voices.Count):
            voice = voices.Item(i)
            name = voice.GetDescription()
            
            # Priorité aux voix françaises
            if any(keyword in name.lower() for keyword in ['french', 'français', 'france', 'fr-']):
                best_voice_idx = i
                best_voice_name = name
                break
            # Sinon, voix de qualité
            elif any(keyword in name.lower() for keyword in ['hazel', 'zira', 'david', 'mark']):
                best_voice_idx = i
                best_voice_name = name
        
        speaker.Voice = voices.Item(best_voice_idx)
        speaker.Rate = 0
        speaker.Volume = 100
        
        print(f"✅ Voix sélectionnée: {best_voice_name}")
        print()
        
        return best_voice_idx, best_voice_name, speaker
        
    except ImportError:
        print("❌ Module win32com non disponible")
        print("💡 Installation: pip install pywin32")
        return None, None, None
    except Exception as e:
        print(f"❌ Erreur SAPI: {e}")
        return None, None, None

def test_assistant_avec_voix_naturelle():
    """Test complet assistant avec voix naturelle sélectionnée"""
    
    # Sélectionner voix naturelle
    voice_idx, voice_name, speaker = test_voix_windows_sapi_amelioree()
    
    if speaker is None:
        print("❌ Aucune voix disponible")
        return False
    
    print("\n🤖 ASSISTANT LUXA - VOIX NATURELLE")
    print("=" * 60)
    print(f"🎤 Voix: {voice_name}")
    print("🎮 GPU: RTX 3090 (CUDA:1)")
    print()
    
    # Messages de présentation
    messages_presentation = [
        "Bonjour ! Je suis LUXA, votre assistant vocal intelligent.",
        "Je fonctionne maintenant avec une voix naturelle de haute qualité.",
        "Ma puissance de calcul provient de votre RTX 3090 avec 24 gigaoctets de mémoire.",
        "Je suis prêt à vous aider avec vos tâches quotidiennes.",
        "Comment puis-je vous assister aujourd'hui ?"
    ]
    
    print("🎯 Messages de présentation:")
    for i, message in enumerate(messages_presentation, 1):
        print(f"   {i}. \"{message}\"")
        print("      🔊 Assistant parle...")
        
        start_time = time.time()
        speaker.Speak(message)
        duration = time.time() - start_time
        
        print(f"      ✅ Synthèse: {duration:.2f}s")
        time.sleep(1)  # Pause naturelle
        print()
    
    # Mode interactif
    print("🗣️ MODE INTERACTIF")
    print("Tapez un message pour l'entendre avec la voix naturelle")
    print("Commandes: 'quit' pour quitter, 'vitesse N' pour changer vitesse (-10 à 10)")
    print()
    
    while True:
        try:
            user_input = input("💬 Votre message: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                speaker.Speak("Au revoir ! J'espère vous avoir satisfait avec ma voix naturelle.")
                print("👋 Au revoir !")
                break
            
            # Commande vitesse
            if user_input.lower().startswith('vitesse '):
                try:
                    vitesse = int(user_input.split()[1])
                    vitesse = max(-10, min(10, vitesse))  # Limiter entre -10 et 10
                    speaker.Rate = vitesse
                    speaker.Speak(f"Vitesse réglée à {vitesse}")
                    print(f"✅ Vitesse modifiée: {vitesse}")
                    continue
                except:
                    print("❌ Format invalide. Utilisez: vitesse N (N entre -10 et 10)")
                    continue
            
            # Messages spéciaux
            if user_input.lower() in ['bonjour', 'salut', 'hello']:
                user_input = "Bonjour ! Ravi de vous entendre. Comment allez-vous aujourd'hui ?"
            elif user_input.lower() in ['merci', 'thank you']:
                user_input = "Je vous en prie ! C'est un plaisir de vous aider avec ma voix naturelle."
            elif user_input.lower() in ['comment ça va', 'ça va']:
                user_input = "Je vais très bien, merci ! Ma voix naturelle fonctionne parfaitement sur votre RTX 3090."
            
            print("🔊 LUXA dit...")
            start_time = time.time()
            speaker.Speak(user_input)
            duration = time.time() - start_time
            
            chars_per_sec = len(user_input) / duration if duration > 0 else 0
            print(f"✅ Synthèse: {duration:.2f}s ({chars_per_sec:.0f} car/s)")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Session interrompue")
            speaker.Speak("Session interrompue. À bientôt !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
            continue
    
    return True

if __name__ == "__main__":
    print("🎤 LUXA - VOIX NATURELLE QUI MARCHE")
    print("🎮 Configuration RTX 3090 (CUDA:1)")
    print()
    
    try:
        success = test_assistant_avec_voix_naturelle()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 VOIX NATURELLE CONFIGURÉE AVEC SUCCÈS !")
            print("✅ LUXA peut maintenant parler avec une voix de qualité")
            print("🎮 RTX 3090 optimisée")
        else:
            print("🚨 PROBLÈME CONFIGURATION VOIX")
            print("⚠️ Vérification système requise")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n👋 Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc() 