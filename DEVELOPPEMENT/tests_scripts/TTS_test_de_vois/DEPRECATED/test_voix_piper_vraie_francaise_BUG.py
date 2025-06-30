#!/usr/bin/env python3
"""
TEST VOIX PIPER FRANÇAISE - VRAI CHEMIN D:\TTS_Voices
🚨 RTX 3090 (CUDA:1) - VRAIES VOIX FRANÇAISES

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

def test_piper_siwis_francais():
    """Test voix Piper fr_FR-siwis-medium avec VRAI CHEMIN"""
    
    print("\n🎭 TEST PIPER SIWIS FRANÇAIS (VRAI CHEMIN)")
    print("=" * 50)
    
    try:
        sys.path.append('TTS')
        from tts_handler_piper_french import TTSHandlerPiperFrench
        import sounddevice as sd
        import numpy as np
        
        # ✅ VRAI CHEMIN D:\TTS_Voices\piper\!
        config = {
            'model_path': r'D:\TTS_Voices\piper\fr_FR-siwis-medium.onnx',
            'config_path': r'D:\TTS_Voices\piper\fr_FR-siwis-medium.onnx.json',
            'use_gpu': True,
            'device': 'cuda:1'  # RTX 3090
        }
        
        print(f"📁 Modèle: {config['model_path']}")
        print(f"📄 Config: {config['config_path']}")
        
        # Vérification fichiers
        if not os.path.exists(config['model_path']):
            print(f"❌ Modèle introuvable: {config['model_path']}")
            return False
            
        if not os.path.exists(config['config_path']):
            print(f"❌ Config introuvable: {config['config_path']}")
            return False
            
        print("✅ Fichiers trouvés")
        
        # Initialisation
        print("🚀 Initialisation Piper Siwis...")
        handler = TTSHandlerPiperFrench(config)
        print("✅ Handler initialisé")
        
        # Texte français
        texte = "Bonjour ! Je suis LUXA, votre assistant vocal français intelligent. J'utilise une voix française naturelle et je peux vous aider avec toutes vos tâches."
        print(f"📝 Texte: {texte[:50]}...")
        
        # Synthèse
        print("🎤 Synthèse vocale française...")
        audio_data = handler.synthesize(texte)
        
        if audio_data is not None and len(audio_data) > 0:
            print(f"✅ Audio généré: {len(audio_data)} échantillons")
            print(f"🔊 Amplitude: {np.max(np.abs(audio_data)):.3f}")
            
            # Lecture
            print("🔊 Lecture audio française...")
            sd.play(audio_data, samplerate=22050)
            sd.wait()
            print("✅ Lecture française terminée")
            
            return True
        else:
            print("❌ Échec génération audio")
            return False
            
    except Exception as e:
        print(f"❌ Erreur Piper Siwis: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_piper_alternatifs():
    """Test d'autres voix Piper disponibles"""
    
    print("\n🎭 TEST VOIX PIPER ALTERNATIVES")
    print("=" * 40)
    
    modeles = [
        'fr_FR-mls_1840-medium.onnx',
        'fr_FR-upmc-medium.onnx'
    ]
    
    for modele in modeles:
        chemin = rf'D:\TTS_Voices\piper\{modele}'
        print(f"\n📁 Test: {modele}")
        
        if os.path.exists(chemin):
            taille = os.path.getsize(chemin)
            print(f"📊 Taille: {taille} octets")
            
            if taille < 100:
                print("⚠️ Fichier trop petit (probablement corrompu)")
            else:
                print("✅ Fichier valide")
        else:
            print("❌ Fichier introuvable")

if __name__ == "__main__":
    print("🎯 DÉMARRAGE TEST VOIX FRANÇAISES PIPER")
    print("=" * 60)
    
    # Test principal
    succes = test_piper_siwis_francais()
    
    # Tests alternatifs
    test_piper_alternatifs()
    
    # Résultat
    if succes:
        print("\n🎉 SUCCÈS : Voix française Piper fonctionnelle !")
    else:
        print("\n❌ ÉCHEC : Problème avec la voix française")
    
    print("\n🏁 FIN DES TESTS") 