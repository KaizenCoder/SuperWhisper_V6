#!/usr/bin/env python3
"""
Test TTS Piper avec textes longs pour feedback qualité vocale
Évaluation complète de la compréhensibilité et prosodie

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

import time

# Configuration RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tts_feedback_long():
    """Test TTS avec textes longs pour feedback qualité"""
    
    print("🎤 TEST TTS PIPER - FEEDBACK QUALITÉ VOCALE")
    print("=" * 60)
    
    # Configuration
    config = {
        'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
        'config_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json',
        'use_gpu': True
    }
    
    try:
        # Import et initialisation
        print("1. 🚀 Initialisation handler RTX 3090...")
        from TTS.tts_handler_piper_fixed import TTSHandlerPiperFixed
        
        start_time = time.time()
        handler = TTSHandlerPiperFixed(config)
        init_time = time.time() - start_time
        print(f"✅ Handler initialisé en {init_time:.2f}s")
        
        # Tests de qualité progressive
        test_scenarios = [
            {
                "name": "📖 PRÉSENTATION LUXA",
                "text": "Bonjour ! Je suis LUXA, votre assistant vocal intelligent. Je peux vous aider avec diverses tâches grâce à mes capacités d'intelligence artificielle avancées. Mon système fonctionne entièrement en local pour garantir votre confidentialité et votre sécurité."
            },
            {
                "name": "🔬 DESCRIPTION TECHNIQUE", 
                "text": "LUXA utilise un pipeline sophistiqué combinant reconnaissance vocale, traitement par modèle de langage, et synthèse vocale. Le système s'appuie sur des technologies comme Whisper pour la transcription, des modèles Llama pour la compréhension, et Piper pour la génération audio haute qualité."
            },
            {
                "name": "📚 LECTURE NARRATIVE",
                "text": "Dans un futur proche, les assistants vocaux intelligents révolutionnent notre quotidien. Ils comprennent le langage naturel, analysent le contexte, et répondent de manière pertinente. Cette technologie représente une avancée majeure dans l'interaction homme-machine, ouvrant de nouvelles possibilités pour l'accessibilité et la productivité."
            },
            {
                "name": "🎯 INSTRUCTIONS COMPLEXES",
                "text": "Pour configurer votre environnement de développement, vous devez d'abord installer Python trois point douze, puis créer un environnement virtuel. Ensuite, installez les dépendances requises, notamment PyTorch avec support CUDA, onnxruntime pour l'inférence GPU, et les bibliothèques audio comme sounddevice et soundfile. Vérifiez que votre carte graphique RTX 3090 est correctement détectée."
            },
            {
                "name": "🗣️ CONVERSATION NATURELLE",
                "text": "Vous savez, l'intelligence artificielle a beaucoup évolué ces dernières années. Ce qui me fascine le plus, c'est la capacité des modèles modernes à comprendre les nuances du langage humain. Par exemple, ils peuvent détecter l'ironie, l'émotion, et même s'adapter au style de conversation de leur interlocuteur. C'est vraiment impressionnant, vous ne trouvez pas ?"
            }
        ]
        
        print(f"\n2. 🎭 Tests de qualité vocale ({len(test_scenarios)} scénarios)")
        print("   💡 Écoutez attentivement chaque test pour évaluer :")
        print("      - Clarté de la prononciation")
        print("      - Fluidité et rythme") 
        print("      - Intonation naturelle")
        print("      - Compréhensibilité globale")
        
        total_chars = 0
        total_time = 0
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\n" + "─" * 60)
            print(f"🎯 TEST {i+1}/5: {scenario['name']}")
            print(f"📝 Texte ({len(scenario['text'])} caractères):")
            print(f"   \"{scenario['text'][:100]}...\"")
            
            # Pause avant synthèse
            input(f"\n⏸️ Appuyez sur ENTRÉE pour démarrer la synthèse...")
            
            # Synthèse et lecture
            start_time = time.time()
            print(f"🔊 Synthèse en cours...")
            
            audio_data = handler.synthesize(scenario['text'])
            synth_time = time.time() - start_time
            
            if len(audio_data) > 0:
                # Calculs de performance
                chars_per_sec = len(scenario['text']) / synth_time
                audio_duration = len(audio_data) / handler.sample_rate
                rtf = synth_time / audio_duration
                
                print(f"   ⚡ Synthèse: {synth_time:.2f}s ({chars_per_sec:.0f} car/s)")
                print(f"   🎵 Durée audio: {audio_duration:.1f}s")
                print(f"   🚀 RTF: {rtf:.3f}")
                print(f"   🔊 LECTURE EN COURS...")
                
                # Lecture audio
                handler.speak(scenario['text'])
                
                # Feedback utilisateur
                print(f"\n📊 Évaluation de la qualité :")
                print(f"   1️⃣ Excellent  2️⃣ Bon  3️⃣ Moyen  4️⃣ Faible")
                feedback = input(f"   Votre note (1-4) : ")
                
                # Commentaires optionnels
                comments = input(f"   Commentaires (optionnel) : ")
                
                print(f"   ✅ Feedback enregistré : {feedback}/4")
                if comments:
                    print(f"   💬 Commentaire : \"{comments}\"")
                
                total_chars += len(scenario['text'])
                total_time += synth_time
                
            else:
                print(f"   ❌ Échec synthèse")
                
            print(f"─" * 60)
        
        # Résumé final
        if total_time > 0:
            avg_chars_per_sec = total_chars / total_time
            print(f"\n📊 RÉSUMÉ PERFORMANCE GLOBALE:")
            print(f"   📝 Total caractères: {total_chars:,}")
            print(f"   ⏱️ Temps total synthèse: {total_time:.1f}s")
            print(f"   ⚡ Performance moyenne: {avg_chars_per_sec:.0f} caractères/s")
            print(f"   🎮 GPU RTX 3090: {'✅ Actif' if 'CUDA' in str(handler.session.get_providers()) else '❌ CPU'}")
            
        print(f"\n🎉 TEST QUALITÉ VOCALE TERMINÉ")
        print(f"💡 Ce feedback aidera à optimiser la qualité de LUXA")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎤 LUXA - TEST QUALITÉ VOCALE APPROFONDI")
    print("🎯 Objectif: Évaluer la compréhensibilité et naturalité")
    print("⏱️ Durée estimée: 5-10 minutes")
    print()
    
    proceed = input("▶️ Continuer le test complet ? (o/n): ")
    if proceed.lower() in ['o', 'oui', 'y', 'yes']:
        success = test_tts_feedback_long()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 TEST QUALITÉ VOCALE COMPLÉTÉ")
            print("💡 Merci pour votre feedback sur la voix de LUXA !")
        else:
            print("🚨 PROBLÈME DURANT LE TEST")
        print("=" * 60)
    else:
        print("⏹️ Test annulé") 