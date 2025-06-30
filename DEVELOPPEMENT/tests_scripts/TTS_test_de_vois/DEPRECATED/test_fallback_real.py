#!/usr/bin/env python3
"""
Test pratique du système de fallback avec simulation de pannes.

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
import os
import sys

# Configuration GPU RTX 3090 obligatoire
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Import du manager unifié
sys.path.append('.')
from TTS.tts_manager import UnifiedTTSManager

async def test_fallback_simulation():
    print("🔧 TEST FALLBACK RÉEL - Simulation pannes")
    
    with open('config/tts.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test 1: Tous handlers actifs
    print("\n1️⃣ Test normal (tous handlers actifs)")
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test normal avec tous les backends.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    # Test 2: Désactiver piper_native (forcer fallback)
    print("\n2️⃣ Test fallback (piper_native désactivé)")
    config['backends']['piper_native']['enabled'] = False
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test fallback vers piper CLI.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    # Test 3: Désactiver piper_native + piper_cli (forcer SAPI)
    print("\n3️⃣ Test fallback SAPI (piper désactivés)")
    config['backends']['piper_cli']['enabled'] = False
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test fallback vers SAPI français.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    # Test 4: Tous désactivés sauf emergency
    print("\n4️⃣ Test emergency (tous backends désactivés)")
    config['backends']['sapi_french']['enabled'] = False
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test handler d'urgence silencieux.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    print(f"\n🎯 VALIDATION: Chaîne de fallback complète testée!")

if __name__ == "__main__":
    asyncio.run(test_fallback_simulation()) 