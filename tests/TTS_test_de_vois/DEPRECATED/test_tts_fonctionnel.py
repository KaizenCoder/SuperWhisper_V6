#!/usr/bin/env python3
"""
Test fonctionnel complet du système TTS Piper
Synthèse vocale réelle avec modèle français

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
import traceback

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_piper_synthesis():
    """Test de synthèse vocale complète avec Piper"""
    
    print("🎯 Test fonctionnel TTS Piper")
    print("=" * 50)
    
    # Configuration du test
    config = {
        'model_path': './models/fr_FR-siwis-medium.onnx',
        'use_gpu': True
    }
    
    # Texte de test en français
    test_text = "Bonjour ! Je suis LUXA, votre assistant vocal intelligent. Ceci est un test de synthèse vocale française avec Piper."
    
    try:
        # Étape 1: Vérification du modèle
        print("\n1. 🔍 Vérification du modèle...")
        if not os.path.exists(config['model_path']):
            print(f"❌ Modèle non trouvé: {config['model_path']}")
            return False
        
        file_size = os.path.getsize(config['model_path']) / (1024*1024)  # MB
        print(f"✅ Modèle trouvé: {config['model_path']} ({file_size:.1f} MB)")
        
        # Étape 2: Importation du handler
        print("\n2. 📦 Chargement du handler TTS...")
        from TTS.tts_handler_piper import TTSHandlerPiper
        print("✅ Handler importé avec succès")
        
        # Étape 3: Initialisation
        print("\n3. 🚀 Initialisation du système TTS...")
        start_time = time.time()
        
        tts_handler = TTSHandlerPiper(config)
        
        init_time = time.time() - start_time
        print(f"✅ Système initialisé en {init_time:.2f}s")
        
        # Étape 4: Test de synthèse
        print("\n4. 🔊 Test de synthèse vocale...")
        print(f"Texte: '{test_text}'")
        
        synthesis_start = time.time()
        
        # Appel de la synthèse
        tts_handler.speak(test_text)
        
        synthesis_time = time.time() - synthesis_start
        print(f"✅ Synthèse terminée en {synthesis_time:.2f}s")
        
        # Étape 5: Résultats
        print("\n5. 📊 Résultats du test:")
        print(f"   • Temps d'initialisation: {init_time:.2f}s")
        print(f"   • Temps de synthèse: {synthesis_time:.2f}s")
        print(f"   • Longueur du texte: {len(test_text)} caractères")
        print(f"   • Performance: {len(test_text)/synthesis_time:.1f} caractères/s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print(f"\n🔍 Trace complète:")
        traceback.print_exc()
        return False

def test_advanced_features():
    """Test des fonctionnalités avancées"""
    
    print("\n" + "=" * 50)
    print("🧪 Tests avancés")
    print("=" * 50)
    
    config = {
        'model_path': './models/fr_FR-siwis-medium.onnx',
        'use_gpu': True
    }
    
    try:
        from TTS.tts_handler_piper import TTSHandlerPiper
        tts_handler = TTSHandlerPiper(config)
        
        # Test 1: Texte court
        print("\n1. Test texte court:")
        short_text = "Salut !"
        start = time.time()
        tts_handler.speak(short_text)
        print(f"   ✅ Texte court ({len(short_text)} chars) en {time.time()-start:.2f}s")
        
        # Test 2: Texte long
        print("\n2. Test texte long:")
        long_text = "Ceci est un test de synthèse vocale avec un texte plus long pour évaluer les performances sur plusieurs phrases. LUXA est un assistant vocal intelligent qui utilise des technologies avancées pour fournir une expérience utilisateur de haute qualité."
        start = time.time()
        tts_handler.speak(long_text)
        print(f"   ✅ Texte long ({len(long_text)} chars) en {time.time()-start:.2f}s")
        
        # Test 3: Caractères spéciaux
        print("\n3. Test caractères spéciaux:")
        special_text = "Test avec des chiffres 123, des signes ! ? @ # et des accents: éàèùôî"
        start = time.time()
        tts_handler.speak(special_text)
        print(f"   ✅ Caractères spéciaux en {time.time()-start:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests avancés: {e}")
        return False

def main():
    """Fonction principale du test"""
    
    print("🎤 TEST FONCTIONNEL COMPLET - SYSTÈME TTS PIPER")
    print("=" * 60)
    
    # Test de base
    success = test_piper_synthesis()
    
    if success:
        print("\n✅ Test de base réussi !")
        
        # Tests avancés si le test de base passe
        advanced_success = test_advanced_features()
        
        if advanced_success:
            print("\n🎉 TOUS LES TESTS RÉUSSIS !")
            print("Le système TTS Piper est entièrement fonctionnel.")
        else:
            print("\n⚠️ Tests de base OK, mais problèmes avec les tests avancés")
    else:
        print("\n❌ Échec du test de base")
        print("Vérifiez la configuration et les dépendances")
    
    print("\n" + "=" * 60)
    print("Fin des tests")

if __name__ == "__main__":
    main() 