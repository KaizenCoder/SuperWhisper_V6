#!/usr/bin/env python3
"""
Benchmark de performance avec mesures réelles et validation des KPI.

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
import statistics
import yaml
import os
import sys

# Configuration GPU RTX 3090 obligatoire
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Import du manager unifié
sys.path.append('.')
from TTS.tts_manager import UnifiedTTSManager

async def benchmark_performance():
    print("⚡ BENCHMARK PERFORMANCE RÉEL")
    print("=" * 50)
    
    with open('config/tts.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    manager = UnifiedTTSManager(config)
    
    # Phrases de test de différentes longueurs
    test_cases = [
        ("Court", "Bonjour."),
        ("Moyen", "Bonjour, je suis votre assistant vocal SuperWhisper."),
        ("Long", "Bonjour, je suis votre assistant vocal SuperWhisper. La synthèse vocale fonctionne parfaitement avec la carte graphique RTX 3090 et les modèles Piper français."),
    ]
    
    results = {}
    
    for case_name, text in test_cases:
        print(f"\n📊 Test {case_name}: '{text[:40]}...'")
        latencies = []
        
        # 10 mesures pour statistiques fiables
        for i in range(10):
            start_time = time.perf_counter()
            result = await manager.synthesize(text)
            latency = (time.perf_counter() - start_time) * 1000
            
            if result.success:
                latencies.append(latency)
                print(f"  Run {i+1:2d}: {latency:6.1f}ms ({result.backend_used})")
            else:
                print(f"  Run {i+1:2d}: ÉCHEC - {result.error}")
        
        if latencies:
            results[case_name] = {
                'mean': statistics.mean(latencies),
                'p95': sorted(latencies)[int(0.95 * len(latencies))],
                'min': min(latencies),
                'max': max(latencies),
                'backend': result.backend_used
            }
    
    # Rapport final
    print(f"\n🎯 RAPPORT PERFORMANCE FINALE")
    print("=" * 50)
    for case_name, stats in results.items():
        print(f"{case_name:6s}: Moy={stats['mean']:6.1f}ms | P95={stats['p95']:6.1f}ms | Min={stats['min']:6.1f}ms | Max={stats['max']:6.1f}ms | Backend={stats['backend']}")
    
    # Validation KPI
    print(f"\n✅ VALIDATION KPI:")
    if 'Court' in results:
        p95_court = results['Court']['p95']
        backend = results['Court']['backend']
        if backend == 'piper_native' and p95_court < 120:
            print(f"✅ PiperNative P95: {p95_court:.1f}ms < 120ms TARGET")
        elif backend == 'piper_cli' and p95_court < 1000:
            print(f"✅ PiperCLI P95: {p95_court:.1f}ms < 1000ms TARGET")
        elif backend == 'sapi_french' and p95_court < 2000:
            print(f"✅ SAPI P95: {p95_court:.1f}ms < 2000ms TARGET")
        else:
            print(f"⚠️ Performance: {backend} P95={p95_court:.1f}ms")

if __name__ == "__main__":
    asyncio.run(benchmark_performance()) 