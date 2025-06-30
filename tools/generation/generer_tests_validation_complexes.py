#!/usr/bin/env python3
"""
Génération tests validation complexes - SuperWhisper V6 TTS
Génère des fichiers audio avec textes de validation de complexité croissante

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
from TTS.tts_manager import UnifiedTTSManager
from TTS.utils_audio import is_valid_wav, get_wav_info

# Textes de validation complexes
TEXTE_VALIDATION_1 = """Bonjour, ceci est un test de validation pour SuperWhisper2. Je vais maintenant énoncer plusieurs phrases de complexité croissante pour évaluer la précision de transcription.
Premièrement, des mots simples : chat, chien, maison, voiture, ordinateur, téléphone.
Deuxièmement, des phrases courtes : Il fait beau aujourd'hui. Le café est délicieux. J'aime la musique classique.
Troisièmement, des phrases plus complexes : L'intelligence artificielle transforme notre manière de travailler et de communiquer dans le monde moderne.
Quatrièmement, des termes techniques : algorithme, machine learning, GPU RTX 3090, faster-whisper, quantification INT8, latence de transcription."""

TEXTE_VALIDATION_2 = """Cinquièmement, des nombres et dates : vingt-trois, quarante-sept, mille neuf cent quatre-vingt-quinze, le quinze janvier deux mille vingt-quatre.
Sixièmement, des mots difficiles : chrysanthème, anticonstitutionnellement, prestidigitateur, kakémono, yaourt.
Septièmement, une phrase longue et complexe : L'optimisation des performances de transcription vocale nécessite une approche méthodique combinant la sélection appropriée des modèles, l'ajustement des paramètres de traitement, et l'implémentation d'algorithmes de post-traitement pour améliorer la qualité du résultat final.
Fin du test de validation."""

async def generer_fichier_tts(manager, texte, nom_fichier, backend_specifique=None):
    """Génère un fichier TTS avec validation complète"""
    print(f"\n🎯 Génération: {nom_fichier}")
    print("-" * 50)
    
    try:
        start_time = time.perf_counter()
        
        if backend_specifique:
            # Test backend spécifique
            handler = manager.handlers.get(backend_specifique)
            if not handler:
                print(f"❌ Backend {backend_specifique.value} non disponible")
                return False
            
            audio_data = await handler.synthesize(texte)
            backend_used = backend_specifique.value
        else:
            # Test manager unifié
            result = await manager.synthesize(texte)
            if not result.success:
                print(f"❌ Échec synthèse: {result.error}")
                return False
            
            audio_data = result.audio_data
            backend_used = result.backend_used
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Validation format
        if not is_valid_wav(audio_data):
            print(f"❌ Format WAV invalide")
            return False
        
        wav_info = get_wav_info(audio_data)
        
        # Sauvegarde fichier
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / nom_fichier
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        # Affichage résultats
        print(f"✅ SUCCÈS")
        print(f"   Backend: {backend_used}")
        print(f"   Latence: {latency_ms:.1f}ms")
        print(f"   Taille: {len(audio_data)} bytes ({len(audio_data)/1024:.1f}KB)")
        
        if 'error' not in wav_info:
            print(f"   Durée: {wav_info.get('duration_ms', 'N/A')}ms")
            print(f"   Qualité: {wav_info.get('framerate', 'N/A')} Hz, {wav_info.get('channels', 'N/A')} canaux")
        
        print(f"   Fichier: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return False

async def generer_tests_validation_complexes():
    """Génère tous les tests de validation complexes"""
    print("\n🧪 GÉNÉRATION TESTS VALIDATION COMPLEXES - SUPERWHISPER V6")
    print("=" * 70)
    
    # Chargement configuration
    config_path = Path("config/tts.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialisation manager
    print("📋 Initialisation UnifiedTTSManager...")
    manager = UnifiedTTSManager(config)
    
    print(f"🎯 Backends disponibles: {len(manager.handlers)}")
    for backend_type in manager.handlers.keys():
        print(f"   - {backend_type.value}")
    
    # Tests à générer
    tests = [
        # Tests avec manager unifié (fallback automatique)
        {
            'texte': TEXTE_VALIDATION_1,
            'nom': 'validation_complexe_partie1_unifie.wav',
            'backend': None,
            'description': 'Partie 1 - Manager unifié (mots simples → termes techniques)'
        },
        {
            'texte': TEXTE_VALIDATION_2,
            'nom': 'validation_complexe_partie2_unifie.wav',
            'backend': None,
            'description': 'Partie 2 - Manager unifié (nombres → phrase complexe)'
        },
        
        # Tests avec backends spécifiques
        {
            'texte': TEXTE_VALIDATION_1,
            'nom': 'validation_complexe_partie1_piper_native.wav',
            'backend': 'piper_native',
            'description': 'Partie 1 - Piper Native GPU'
        },
        {
            'texte': TEXTE_VALIDATION_2,
            'nom': 'validation_complexe_partie2_piper_native.wav',
            'backend': 'piper_native',
            'description': 'Partie 2 - Piper Native GPU'
        },
        {
            'texte': TEXTE_VALIDATION_1,
            'nom': 'validation_complexe_partie1_piper_cli.wav',
            'backend': 'piper_cli',
            'description': 'Partie 1 - Piper CLI CPU'
        },
        {
            'texte': TEXTE_VALIDATION_2,
            'nom': 'validation_complexe_partie2_piper_cli.wav',
            'backend': 'piper_cli',
            'description': 'Partie 2 - Piper CLI CPU'
        },
        {
            'texte': TEXTE_VALIDATION_1,
            'nom': 'validation_complexe_partie1_sapi.wav',
            'backend': 'sapi_french',
            'description': 'Partie 1 - SAPI French'
        },
        {
            'texte': TEXTE_VALIDATION_2,
            'nom': 'validation_complexe_partie2_sapi.wav',
            'backend': 'sapi_french',
            'description': 'Partie 2 - SAPI French'
        }
    ]
    
    # Statistiques
    total_tests = len(tests)
    success_count = 0
    error_count = 0
    
    print(f"\n🎯 Génération de {total_tests} fichiers de test...")
    print("=" * 70)
    
    # Génération de tous les tests
    for i, test in enumerate(tests, 1):
        print(f"\n📋 TEST {i}/{total_tests}: {test['description']}")
        print(f"📝 Texte: {len(test['texte'])} caractères")
        
        # Conversion backend string vers enum si nécessaire
        backend_enum = None
        if test['backend']:
            from TTS.tts_manager import TTSBackendType
            backend_map = {
                'piper_native': TTSBackendType.PIPER_NATIVE,
                'piper_cli': TTSBackendType.PIPER_CLI,
                'sapi_french': TTSBackendType.SAPI_FRENCH,
                'silent_emergency': TTSBackendType.SILENT_EMERGENCY
            }
            backend_enum = backend_map.get(test['backend'])
        
        # Génération
        success = await generer_fichier_tts(
            manager, 
            test['texte'], 
            test['nom'], 
            backend_enum
        )
        
        if success:
            success_count += 1
        else:
            error_count += 1
    
    # Génération fichier complet (combiné)
    print(f"\n🎯 GÉNÉRATION FICHIER COMPLET COMBINÉ")
    print("-" * 50)
    
    texte_complet = TEXTE_VALIDATION_1 + "\n\n" + TEXTE_VALIDATION_2
    success_complet = await generer_fichier_tts(
        manager,
        texte_complet,
        'validation_complexe_complet.wav'
    )
    
    if success_complet:
        success_count += 1
        total_tests += 1
    else:
        error_count += 1
        total_tests += 1
    
    # Résumé final
    print(f"\n📊 RÉSUMÉ GÉNÉRATION TESTS VALIDATION")
    print("=" * 70)
    print(f"Tests générés avec succès: {success_count}/{total_tests}")
    print(f"Erreurs: {error_count}")
    print(f"Taux de réussite: {success_count/total_tests*100:.1f}%")
    
    if success_count > 0:
        print(f"\n🎵 FICHIERS GÉNÉRÉS DISPONIBLES:")
        output_dir = Path("test_output")
        validation_files = sorted(output_dir.glob("validation_complexe_*.wav"))
        
        for wav_file in validation_files:
            try:
                with open(wav_file, 'rb') as f:
                    data = f.read()
                
                if is_valid_wav(data):
                    wav_info = get_wav_info(data)
                    duration = wav_info.get('duration_ms', 0) if 'error' not in wav_info else 0
                    size_mb = len(data) / 1024 / 1024
                    
                    print(f"✅ {wav_file.name:40} | {duration/1000:5.1f}s | {size_mb:6.2f}MB")
                else:
                    print(f"❌ {wav_file.name:40} | FORMAT INVALIDE")
                    
            except Exception as e:
                print(f"❌ {wav_file.name:40} | ERREUR: {e}")
    
    # Nettoyage
    await manager.cleanup()
    
    return success_count, error_count

if __name__ == "__main__":
    async def main():
        print("🚀 DÉMARRAGE GÉNÉRATION TESTS VALIDATION COMPLEXES")
        
        success, errors = await generer_tests_validation_complexes()
        
        print(f"\n🏁 GÉNÉRATION TERMINÉE")
        print("=" * 70)
        
        if success > 0:
            print(f"✅ {success} fichiers générés avec succès!")
            print("🎵 Vous pouvez maintenant écouter les tests de validation complexes!")
            print("📁 Fichiers disponibles dans: test_output/validation_complexe_*.wav")
        
        if errors > 0:
            print(f"⚠️  {errors} erreurs détectées")
        
        print("\n🎯 FICHIERS RECOMMANDÉS POUR ÉCOUTE:")
        print("1. validation_complexe_complet.wav (texte complet)")
        print("2. validation_complexe_partie1_*.wav (mots simples → techniques)")
        print("3. validation_complexe_partie2_*.wav (nombres → phrase complexe)")
    
    asyncio.run(main()) 