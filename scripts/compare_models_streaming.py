#!/usr/bin/env python3
"""
Comparaison modèles streaming - SuperWhisper V6
Compare la précision et latence des différents modèles

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
import time
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 Comparaison Modèles Streaming - Configuration GPU RTX 3090 (CUDA:1)")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports après configuration GPU
import torch
from STT.unified_stt_manager import UnifiedSTTManager
from STT.streaming_microphone_manager import StreamingMicrophoneManager

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
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

class ModelComparator:
    """Comparateur de modèles pour streaming"""
    
    def __init__(self):
        self.results = {}
    
    async def test_model(self, model_name: str, config: dict, duration: int = 30):
        """Test un modèle spécifique"""
        print(f"\n🧪 Test modèle: {model_name}")
        print(f"⏱️ Durée: {duration}s")
        print("-" * 40)
        
        try:
            # Initialisation
            start_init = time.time()
            stt_mgr = UnifiedSTTManager(config)
            mic_mgr = StreamingMicrophoneManager(stt_mgr)
            init_time = time.time() - start_init
            
            print(f"✅ Initialisation: {init_time:.1f}s")
            print("🎤 Parlez maintenant...")
            
            # Collecte des métriques
            transcriptions = []
            latencies = []
            
            # Hook pour capturer les résultats
            original_callback = mic_mgr._on_transcription
            def capture_callback(result):
                transcriptions.append(result.text)
                latencies.append(result.latency_ms)
                print(f"📝 [{result.latency_ms:.0f}ms] {result.text}")
                original_callback(result)
            
            mic_mgr._on_transcription = capture_callback
            
            # Test avec timeout
            try:
                await asyncio.wait_for(mic_mgr.run(), timeout=duration)
            except asyncio.TimeoutError:
                print(f"⏰ Test terminé après {duration}s")
            
            # Calcul des métriques
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            total_transcriptions = len(transcriptions)
            
            self.results[model_name] = {
                'init_time': init_time,
                'avg_latency': avg_latency,
                'transcriptions': total_transcriptions,
                'texts': transcriptions
            }
            
            print(f"📊 Résultats {model_name}:")
            print(f"   Initialisation: {init_time:.1f}s")
            print(f"   Latence moyenne: {avg_latency:.0f}ms")
            print(f"   Transcriptions: {total_transcriptions}")
            
        except Exception as e:
            print(f"❌ Erreur {model_name}: {e}")
            self.results[model_name] = {'error': str(e)}
    
    def print_comparison(self):
        """Affiche la comparaison finale"""
        print("\n" + "="*60)
        print("📊 COMPARAISON FINALE DES MODÈLES")
        print("="*60)
        
        for model_name, results in self.results.items():
            if 'error' in results:
                print(f"❌ {model_name}: ERREUR - {results['error']}")
            else:
                print(f"✅ {model_name}:")
                print(f"   🚀 Init: {results['init_time']:.1f}s")
                print(f"   ⚡ Latence: {results['avg_latency']:.0f}ms")
                print(f"   📝 Transcriptions: {results['transcriptions']}")
        
        print("\n🎯 RECOMMANDATIONS:")
        print("• small: Rapide mais imprécis (tests/démo)")
        print("• medium: Bon compromis vitesse/précision")
        print("• large-v2: Précision maximale (production)")

async def main():
    """Test comparatif des modèles"""
    print("🚀 Comparaison Modèles Streaming - SuperWhisper V6")
    print("=" * 60)
    
    # Validation GPU RTX 3090
    try:
        validate_rtx3090_configuration()
    except Exception as e:
        print(f"❌ Erreur configuration GPU: {e}")
        return
    
    comparator = ModelComparator()
    
    # Configurations des modèles à tester
    models_to_test = {
        'small': {
            'fallback_chain': ['prism_small'],
            'cache_size_mb': 50,
            'cache_ttl': 300,
            'timeout_per_minute': 2.0,
            'retry_attempts': 2,
            'enable_fallback': True
        },
        'medium': {
            'fallback_chain': ['prism_medium'],
            'cache_size_mb': 100,
            'cache_ttl': 450,
            'timeout_per_minute': 3.0,
            'retry_attempts': 2,
            'enable_fallback': True
        },
        'large-v2': {
            'fallback_chain': ['prism_large_v2'],
            'cache_size_mb': 200,
            'cache_ttl': 600,
            'timeout_per_minute': 5.0,
            'retry_attempts': 3,
            'enable_fallback': False
        }
    }
    
    print("\n🎯 Tests disponibles:")
    for i, model in enumerate(models_to_test.keys(), 1):
        print(f"  {i}. {model}")
    print("  4. Tous les modèles (séquentiel)")
    
    try:
        choice = input("\n🔢 Choisissez un test (1-4): ").strip()
        
        if choice == "4":
            # Test tous les modèles
            for model_name, config in models_to_test.items():
                await comparator.test_model(model_name, config, duration=20)
                if model_name != list(models_to_test.keys())[-1]:
                    print("\n⏳ Pause 5s entre les modèles...")
                    await asyncio.sleep(5)
        else:
            # Test modèle spécifique
            model_names = list(models_to_test.keys())
            if choice.isdigit() and 1 <= int(choice) <= 3:
                model_name = model_names[int(choice) - 1]
                config = models_to_test[model_name]
                await comparator.test_model(model_name, config, duration=30)
            else:
                print("❌ Choix invalide")
                return
        
        # Affichage final
        comparator.print_comparison()
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 Démarrage comparaison modèles streaming...")
    
    try:
        asyncio.run(main())
        print("\n✅ Comparaison terminée avec succès")
    except KeyboardInterrupt:
        print("\n🛑 Comparaison interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        sys.exit(1) 