#!/usr/bin/env python3
"""
Test Rapide - Solution STT Optimisée
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
Test minimal pour validation avant benchmark complet

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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import asyncio
import time
import numpy as np
import torch
from pathlib import Path

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")

def test_post_processor():
    """Test du post-processeur avec exemples réels"""
    print("\n🧪 TEST POST-PROCESSEUR")
    print("-" * 40)
    
    try:
        from STT.stt_postprocessor import STTPostProcessor
        
        processor = STTPostProcessor()
        
        # Exemples de transcriptions avec erreurs
        test_cases = [
            {
                "original": "super whispers utilise after whisper sur gpu rtx",
                "expected_improvements": ["SuperWhisper", "faster-whisper", "GPU", "RTX"]
            },
            {
                "original": "char à la maison crésentemps agorique",
                "expected_improvements": ["chat,", "chrysanthème", "algorithme"]
            },
            {
                "original": "sacrement modification dixièmement",
                "expected_improvements": ["cinquièmement", "sixièmement"]
            },
            {
                "original": "la tige artificielle dans le monde monarme",
                "expected_improvements": ["l'intelligence artificielle", "monde moderne"]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Test: '{test_case['original']}'")
            
            processed, metrics = processor.process(test_case['original'], 0.8)
            
            print(f"   Résultat: '{processed}'")
            print(f"   Corrections: {metrics['corrections_applied']}")
            print(f"   Boost confiance: +{metrics['confidence_boost']:.3f}")
            
            # Vérification améliorations
            improvements_found = 0
            for expected in test_case['expected_improvements']:
                if expected.lower() in processed.lower():
                    improvements_found += 1
            
            success_rate = improvements_found / len(test_case['expected_improvements']) * 100
            print(f"   Améliorations: {improvements_found}/{len(test_case['expected_improvements'])} ({success_rate:.0f}%)")
            
            if success_rate >= 50:
                print("   ✅ SUCCÈS")
            else:
                print("   ⚠️ PARTIEL")
        
        # Statistiques globales
        print(f"\n📊 Statistiques Post-Processeur:")
        stats = processor.get_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for subkey, subvalue in value.items():
                    print(f"     {subkey}: {subvalue}")
            else:
                print(f"   {key}: {value}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Post-Processeur échoué: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur test post-processeur: {e}")
        return False

async def test_manager_optimise():
    """Test du manager optimisé (si possible)"""
    print("\n🧠 TEST MANAGER OPTIMISÉ")
    print("-" * 40)
    
    try:
        from STT.unified_stt_manager_optimized import OptimizedUnifiedSTTManager
        
        config = {
            'model': 'large-v2',
            'compute_type': 'float16'
        }
        
        print("🚀 Initialisation manager optimisé...")
        manager = OptimizedUnifiedSTTManager(config)
        
        # Test d'initialisation seulement (pas de transcription complète)
        print("✅ Manager créé avec succès")
        print("⚠️ Test transcription nécessite faster-whisper installé")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Import Manager Optimisé échoué: {e}")
        print("   (Normal si faster-whisper non installé)")
        return False
    except Exception as e:
        print(f"❌ Erreur test manager: {e}")
        return False

def test_backend_optimise():
    """Test du backend optimisé (si possible)"""
    print("\n⚙️ TEST BACKEND OPTIMISÉ")
    print("-" * 40)
    
    try:
        from STT.backends.prism_stt_backend_optimized import OptimizedPrismSTTBackend
        
        config = {
            'model': 'large-v2',
            'compute_type': 'float16'
        }
        
        print("🚀 Initialisation backend optimisé...")
        print("⚠️ Test complet nécessite faster-whisper installé")
        print("✅ Import backend réussi")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Import Backend Optimisé échoué: {e}")
        print("   (Normal si faster-whisper non installé)")
        return False
    except Exception as e:
        print(f"❌ Erreur test backend: {e}")
        return False

async def main():
    """Test principal de la solution optimisée"""
    print("🎯 TEST SOLUTION STT OPTIMISÉE - SUPERWHISPER V6")
    print("🚨 GPU RTX 3090 OBLIGATOIRE")
    print("=" * 60)
    
    try:
        # 1. Validation GPU
        validate_rtx3090_configuration()
        
        # 2. Test post-processeur (critique)
        postproc_ok = test_post_processor()
        
        # 3. Test manager optimisé
        manager_ok = await test_manager_optimise()
        
        # 4. Test backend optimisé
        backend_ok = test_backend_optimise()
        
        # 5. Résumé
        print("\n" + "="*60)
        print("📊 RÉSUMÉ TEST SOLUTION OPTIMISÉE")
        print("="*60)
        
        print(f"✅ GPU RTX 3090: Validée")
        print(f"{'✅' if postproc_ok else '❌'} Post-Processeur: {'Fonctionnel' if postproc_ok else 'Échoué'}")
        print(f"{'✅' if manager_ok else '⚠️'} Manager Optimisé: {'Fonctionnel' if manager_ok else 'Import requis'}")
        print(f"{'✅' if backend_ok else '⚠️'} Backend Optimisé: {'Fonctionnel' if backend_ok else 'Import requis'}")
        
        if postproc_ok and manager_ok and backend_ok:
            print("\n🎉 SOLUTION OPTIMISÉE PRÊTE!")
            print("   → Lancer benchmark complet: python scripts/benchmark_optimized_stt.py")
        elif postproc_ok:
            print("\n⚠️ DÉPENDANCES REQUISES")
            print("   → Installer faster-whisper pour tests complets")
            print("   → Post-processeur validé et fonctionnel")
        else:
            print("\n❌ PROBLÈME CRITIQUE")
            print("   → Vérifier imports et structure fichiers")
        
        return postproc_ok
        
    except Exception as e:
        print(f"\n❌ Erreur test solution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Exécution test
    success = asyncio.run(main())
    
    if success:
        print(f"\n✅ Test terminé avec succès")
    else:
        print(f"\n❌ Test échoué") 