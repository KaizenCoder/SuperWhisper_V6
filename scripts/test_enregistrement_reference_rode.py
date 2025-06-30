#!/usr/bin/env python3
"""
Test Enregistrement Référence Microphone Rode - Validation Correction VAD
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

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

import torch
import time
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime

def validate_rtx3090():
    """Validation GPU RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

def test_enregistrement_reference_rode():
    """Test avec l'enregistrement de référence microphone Rode"""
    
    print("🎤 TEST ENREGISTREMENT RÉFÉRENCE MICROPHONE RODE")
    print("=" * 60)
    
    # Validation GPU
    validate_rtx3090()
    
    # Chemin fichier audio de référence
    audio_file = Path("test_input/enregistrement_avec_lecture_texte_complet_depuis_micro_rode.wav")
    
    if not audio_file.exists():
        print(f"❌ Fichier audio non trouvé: {audio_file}")
        print("📁 Vérifiez que le fichier existe dans test_input/")
        return
    
    print(f"📁 Fichier audio trouvé: {audio_file}")
    
    # Charger audio
    try:
        audio_data, sample_rate = sf.read(str(audio_file))
        print(f"📊 Audio chargé:")
        print(f"   Durée: {len(audio_data) / sample_rate:.2f}s")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Channels: {audio_data.shape}")
        
        # Convertir en mono si stéréo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            print("🔄 Conversion stéréo → mono")
        
        # Resampler à 16kHz si nécessaire
        if sample_rate != 16000:
            import resampy
            audio_data = resampy.resample(audio_data, sample_rate, 16000)
            sample_rate = 16000
            print("🔄 Resampling → 16kHz")
        
        # Normaliser audio
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
        
        print(f"✅ Audio préparé: {len(audio_data)} samples à {sample_rate}Hz")
        
    except Exception as e:
        print(f"❌ Erreur chargement audio: {e}")
        return
    
    # Import du backend STT
    try:
        sys.path.append('.')
        from STT.backends.prism_stt_backend import PrismSTTBackend
        print("✅ Backend STT importé")
    except ImportError as e:
        print(f"❌ Erreur import backend STT: {e}")
        return
    
    # Test transcription avec correction VAD
    try:
        print("\n🎯 DÉBUT TRANSCRIPTION AVEC CORRECTION VAD")
        print("-" * 50)
        
        # Configuration backend
        config = {
            'model': 'large-v2',
            'device': 'cuda:1',
            'compute_type': 'float16',
            'language': 'fr',
            'beam_size': 5,
            'vad_filter': True
        }
        
        # Initialiser backend
        backend = PrismSTTBackend(config)
        print("✅ Backend initialisé")
        
        # Transcription
        start_time = time.perf_counter()
        result = backend._transcribe_sync(audio_data)
        processing_time = time.perf_counter() - start_time
        
        # Calcul métriques
        audio_duration = len(audio_data) / sample_rate
        rtf = processing_time / audio_duration
        
        print(f"✅ Transcription terminée en {processing_time*1000:.0f}ms")
        print(f"📊 RTF: {rtf:.3f}")
        
        # Analyse du texte transcrit
        text = result.get('text', '').strip()
        words = text.split() if text else []
        word_count = len(words)
        
        print(f"\n📝 RÉSULTAT TRANSCRIPTION:")
        print(f"   Texte: '{text}'")
        print(f"   Nombre de mots: {word_count}")
        print(f"   Longueur: {len(text)} caractères")
        
        # Texte de référence attendu (155 mots)
        texte_reference = """Chat, chien, maison, voiture, ordinateur, téléphone. Bonjour, comment allez-vous aujourd'hui ? J'espère que tout va bien pour vous. Le temps est magnifique dehors, le soleil brille et les oiseaux chantent. C'est une journée parfaite pour se promener dans le parc ou pour lire un bon livre sous un arbre. La technologie moderne nous permet de communiquer facilement avec nos proches, même s'ils sont loin de nous. Les smartphones, les ordinateurs et internet ont révolutionné notre façon de vivre et de travailler. Nous pouvons maintenant accéder à une quantité incroyable d'informations en quelques secondes seulement. L'intelligence artificielle progresse rapidement et transforme de nombreux secteurs d'activité. Les voitures autonomes, les assistants vocaux et les systèmes de recommandation font désormais partie de notre quotidien. Il est important de rester curieux et de continuer à apprendre tout au long de notre vie."""
        
        mots_reference = texte_reference.split()
        mots_reference_count = len(mots_reference)
        
        print(f"\n📊 ANALYSE COMPARATIVE:")
        print(f"   Mots attendus: {mots_reference_count}")
        print(f"   Mots transcrits: {word_count}")
        print(f"   Pourcentage: {(word_count / mots_reference_count * 100):.1f}%")
        
        # Évaluation du résultat
        if word_count >= mots_reference_count * 0.9:  # 90% ou plus
            status = "✅ EXCELLENT"
            color = "🟢"
        elif word_count >= mots_reference_count * 0.7:  # 70% ou plus
            status = "✅ BON"
            color = "🟡"
        elif word_count >= mots_reference_count * 0.5:  # 50% ou plus
            status = "⚠️ ACCEPTABLE"
            color = "🟠"
        else:
            status = "❌ INSUFFISANT"
            color = "🔴"
        
        print(f"\n{color} ÉVALUATION: {status}")
        
        # Comparaison avec résultat précédent (25 mots)
        if word_count > 25:
            improvement = ((word_count - 25) / 25) * 100
            print(f"🚀 AMÉLIORATION: +{improvement:.0f}% vs résultat précédent (25 mots)")
        
        # Sauvegarder résultats
        os.makedirs('test_output', exist_ok=True)
        
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "fichier_audio": str(audio_file),
            "duree_audio_s": audio_duration,
            "sample_rate": sample_rate,
            "processing_time_ms": processing_time * 1000,
            "rtf": rtf,
            "texte_transcrit": text,
            "nombre_mots_transcrits": word_count,
            "nombre_mots_attendus": mots_reference_count,
            "pourcentage_completion": (word_count / mots_reference_count * 100),
            "evaluation": status,
            "correction_vad_appliquee": True,
            "backend_utilise": "prism_stt_backend",
            "gpu_utilise": "RTX_3090_CUDA_1"
        }
        
        output_file = 'test_output/validation_enregistrement_rode_reference.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés: {output_file}")
        
        # Résumé final
        print(f"\n🎊 RÉSUMÉ TEST ENREGISTREMENT RODE:")
        print(f"   📁 Fichier: {audio_file.name}")
        print(f"   ⏱️ Durée: {audio_duration:.1f}s")
        print(f"   🎯 Transcription: {word_count}/{mots_reference_count} mots ({(word_count / mots_reference_count * 100):.1f}%)")
        print(f"   ⚡ Performance: {processing_time*1000:.0f}ms (RTF: {rtf:.3f})")
        print(f"   🎮 GPU: RTX 3090 CUDA:1")
        print(f"   {color} Statut: {status}")
        
        return result_data
        
    except Exception as e:
        print(f"❌ Erreur transcription: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🎤 TEST ENREGISTREMENT RÉFÉRENCE MICROPHONE RODE")
    print("🚨 Validation correction VAD avec audio réel")
    print()
    
    result = test_enregistrement_reference_rode()
    
    if result:
        print(f"\n✅ Test terminé avec succès!")
        print(f"📊 Résultat: {result['evaluation']}")
    else:
        print(f"\n❌ Test échoué") 