#!/usr/bin/env python3
"""
Génération fichier complet optimisé - SuperWhisper V6 TTS
Contourne la limitation de 1000 caractères en utilisant SAPI directement

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

import asyncio
import yaml
import time
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Import du système TTS
from TTS.tts_manager import UnifiedTTSManager, TTSBackendType
from TTS.utils_audio import is_valid_wav, get_wav_info

# Texte complet optimisé (version condensée)
TEXTE_COMPLET_OPTIMISE = """Test validation SuperWhisper2 - complexité croissante.
Mots simples : chat, chien, maison, voiture, ordinateur, téléphone.
Phrases courtes : Il fait beau aujourd'hui. Le café est délicieux. J'aime la musique classique.
Phrases complexes : L'intelligence artificielle transforme notre manière de travailler et de communiquer.
Termes techniques : algorithme, machine learning, GPU RTX 3090, faster-whisper, quantification INT8.
Nombres et dates : vingt-trois, quarante-sept, mille neuf cent quatre-vingt-quinze, le quinze janvier deux mille vingt-quatre.
Mots difficiles : chrysanthème, anticonstitutionnellement, prestidigitateur, kakémono, yaourt.
Phrase complexe finale : L'optimisation des performances de transcription vocale nécessite une approche méthodique combinant la sélection appropriée des modèles et l'implémentation d'algorithmes de post-traitement.
Fin du test de validation."""

async def generer_fichier_complet():
    """Génère le fichier complet avec SAPI (pas de limite de caractères)"""
    print("\n🎯 GÉNÉRATION FICHIER COMPLET OPTIMISÉ")
    print("=" * 60)
    
    # Chargement configuration
    config_path = Path("config/tts.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialisation manager
    print("📋 Initialisation UnifiedTTSManager...")
    manager = UnifiedTTSManager(config)
    
    # Test avec SAPI (pas de limite de caractères)
    print(f"📝 Texte: {len(TEXTE_COMPLET_OPTIMISE)} caractères")
    print("🎯 Backend: SAPI French (pas de limite de caractères)")
    
    try:
        start_time = time.perf_counter()
        
        # Utilisation directe du handler SAPI
        sapi_handler = manager.handlers.get(TTSBackendType.SAPI_FRENCH)
        if not sapi_handler:
            print("❌ Handler SAPI non disponible")
            return False
        
        audio_data = await sapi_handler.synthesize(TEXTE_COMPLET_OPTIMISE)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Validation format
        if not is_valid_wav(audio_data):
            print("❌ Format WAV invalide")
            return False
        
        wav_info = get_wav_info(audio_data)
        
        # Sauvegarde fichier
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "validation_complexe_complet_optimise.wav"
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        # Affichage résultats
        print(f"✅ SUCCÈS COMPLET!")
        print(f"   Backend: sapi_french")
        print(f"   Latence: {latency_ms:.1f}ms")
        print(f"   Taille: {len(audio_data)} bytes ({len(audio_data)/1024:.1f}KB)")
        
        if 'error' not in wav_info:
            duration_s = wav_info.get('duration_ms', 0) / 1000
            print(f"   Durée: {duration_s:.1f}s ({wav_info.get('duration_ms', 0)}ms)")
            print(f"   Qualité: {wav_info.get('framerate', 'N/A')} Hz, {wav_info.get('channels', 'N/A')} canaux")
        
        print(f"   Fichier: {output_file}")
        
        # Nettoyage
        await manager.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        await manager.cleanup()
        return False

if __name__ == "__main__":
    async def main():
        print("🚀 GÉNÉRATION FICHIER COMPLET OPTIMISÉ")
        print("=" * 60)
        
        success = await generer_fichier_complet()
        
        if success:
            print(f"\n🎉 FICHIER COMPLET GÉNÉRÉ AVEC SUCCÈS!")
            print("🎵 Vous pouvez maintenant écouter le test de validation complet!")
            print("📁 Fichier: test_output/validation_complexe_complet_optimise.wav")
        else:
            print(f"\n❌ Échec de la génération")
    
    asyncio.run(main()) 