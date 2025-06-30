#!/usr/bin/env python3
"""
Test fonctionnel du nouveau handler TTS Piper Direct

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

def test_piper_direct():
    """Test complet du handler TTS Piper Direct"""
    
    print("🎯 TEST TTS PIPER DIRECT")
    print("=" * 50)
    
    # Configuration du test
    config = {
        'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
        'use_gpu': True,
        'sample_rate': 22050,
        'noise_scale': 0.667,
        'noise_scale_w': 0.8,
        'length_scale': 1.0
    }
    
    try:
        # Vérification du modèle
        print("\n1. 🔍 Vérification du modèle...")
        if not os.path.exists(config['model_path']):
            print(f"❌ Modèle non trouvé: {config['model_path']}")
            return False
        
        file_size = os.path.getsize(config['model_path']) / (1024*1024)  # MB
        print(f"✅ Modèle trouvé: {config['model_path']} ({file_size:.1f} MB)")
        
        # Importation du handler
        print("\n2. 📦 Importation du handler...")
        from TTS.tts_handler_piper_direct import TTSHandlerPiperDirect
        print("✅ Handler importé avec succès")
        
        # Initialisation
        print("\n3. 🚀 Initialisation...")
        start_time = time.time()
        
        tts_handler = TTSHandlerPiperDirect(config)
        
        init_time = time.time() - start_time
        print(f"✅ Handler initialisé en {init_time:.2f}s")
        
        # Test de synthèse courte
        print("\n4. 🔊 Test synthèse courte...")
        short_text = "Salut !"
        
        synthesis_start = time.time()
        tts_handler.speak(short_text)
        synthesis_time = time.time() - synthesis_start
        
        print(f"✅ Synthèse courte en {synthesis_time:.2f}s")
        
        # Test de synthèse longue
        print("\n5. 🔊 Test synthèse longue...")
        long_text = "Bonjour ! Je suis LUXA, votre assistant vocal intelligent. Ceci est un test complet de synthèse vocale française avec Piper Direct."
        
        synthesis_start = time.time()
        tts_handler.speak(long_text)
        synthesis_time = time.time() - synthesis_start
        
        print(f"✅ Synthèse longue en {synthesis_time:.2f}s")
        
        # Résultats
        print("\n6. 📊 Résultats:")
        print(f"   • Temps d'initialisation: {init_time:.2f}s")
        print(f"   • Synthèse courte: {synthesis_time:.2f}s")
        print(f"   • Performance: {len(long_text)/synthesis_time:.1f} caractères/s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print(f"\n🔍 Trace complète:")
        traceback.print_exc()
        return False

def main():
    """Fonction principale"""
    
    print("🎤 TEST FONCTIONNEL TTS PIPER DIRECT")
    print("=" * 60)
    
    success = test_piper_direct()
    
    if success:
        print("\n🎉 TEST RÉUSSI !")
        print("Le système TTS Piper Direct est fonctionnel.")
        print("Vous avez entendu la synthèse vocale française.")
    else:
        print("\n❌ ÉCHEC DU TEST")
        print("Vérifiez les erreurs ci-dessus.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 