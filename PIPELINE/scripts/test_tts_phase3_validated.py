#!/usr/bin/env python3
"""
Test TTS Phase 3 Validée - Configuration SANS SAPI
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Utilise la configuration exacte de la Phase 3 qui avait des performances record :
- PiperNative (GPU RTX 3090) : ~400-500ms
- PiperCLI (CPU) : ~300-400ms  
- Cache LRU : 29.5ms (93.1% hit rate)
- SilentEmergency : ~0.1-0.2ms

SANS SAPI - Configuration validée en production

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
import json
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Imports
sys.path.insert(0, '.')
from TTS.tts_manager import UnifiedTTSManager

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices != '1':
            raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        return True
    except Exception as e:
        print(f"⚠️ Validation GPU échouée: {e}")
        return False

def create_phase3_config():
    """Crée la configuration TTS Phase 3 validée SANS SAPI"""
    return {
        "cache": {
            "enabled": True,
            "max_size_mb": 200,
            "max_entries": 2000,
            "ttl_seconds": 7200,
            "enable_compression": False
        },
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 3,
            "reset_timeout_seconds": 2.0,
            "half_open_max_calls": 1
        },
        "backends": {
            "piper_native_optimized": {
                "enabled": True,
                "target_latency_ms": 80,
                "device": "cuda:1",  # RTX 3090
                "model_path": "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx",
                "priority": 1
            },
            "piper_cli": {
                "enabled": True,
                "target_latency_ms": 1000,
                "priority": 2
            },
            "silent_emergency": {
                "enabled": True,
                "target_latency_ms": 5,
                "priority": 3
            }
        },
        "gpu_optimization": {
            "tts_device": "cuda:1",  # RTX 3090 pour TTS
            "enable_gpu_streams": True,
            "memory_pool_size_mb": 512
        },
        "text_chunking": {
            "enabled": True,
            "max_chunk_length": 800,
            "overlap_chars": 20,
            "speech_rate_cps": 15.0
        },
        "monitoring": {
            "enabled": True,
            "metrics_interval": 5.0,
            "log_level": "INFO"
        },
        "advanced": {
            "max_text_length": 5000,
            "enable_preprocessing": True,
            "enable_postprocessing": True,
            "fallback_on_error": True
        }
    }

async def test_tts_phase3_validated():
    """Test TTS avec configuration Phase 3 validée"""
    print("\n🚀 DÉMARRAGE TEST TTS PHASE 3 VALIDÉE")
    print("=" * 60)
    
    # Validation GPU obligatoire
    if not validate_rtx3090_configuration():
        print("🚫 ÉCHEC: Configuration GPU RTX 3090 invalide")
        return False
    
    try:
        # Configuration Phase 3 validée
        config = create_phase3_config()
        print("✅ Configuration Phase 3 chargée (SANS SAPI)")
        
        # Initialisation TTS Manager
        print("🔧 Initialisation UnifiedTTSManager...")
        tts_manager = UnifiedTTSManager(config)
        
        # Test texte simple
        test_text = "Bonjour, ceci est un test de validation du TTS Phase 3 de SuperWhisper V6."
        print(f"📝 Texte test: {test_text}")
        
        # Synthèse TTS
        print("🎵 Synthèse TTS en cours...")
        start_time = time.time()
        
        result = await tts_manager.synthesize(test_text)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Validation résultat
        if result and result.success and result.audio_data:
            print(f"✅ TTS SUCCÈS!")
            print(f"🎯 Backend utilisé: {result.backend_used}")
            print(f"⚡ Latence: {latency_ms:.1f}ms")
            print(f"🔊 Audio généré: {len(result.audio_data):,} bytes")
            
            # Validation backend (pas SAPI)
            if result.backend_used == 'sapi_french':
                print("🚫 ERREUR: SAPI utilisé alors qu'il est interdit!")
                return False
            
            # Sauvegarde audio pour validation humaine
            output_file = Path("PIPELINE/test_output/tts_phase3_validated.wav")
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'wb') as f:
                f.write(result.audio_data)
            
            print(f"💾 Audio sauvegardé: {output_file}")
            
            # Validation humaine
            print("\n" + "="*60)
            print("🎧 VALIDATION HUMAINE REQUISE")
            print("="*60)
            print(f"📁 Fichier audio: {output_file}")
            print("🔊 Veuillez écouter le fichier audio généré.")
            print("❓ Entendez-vous une vraie voix française (pas un bip) ?")
            
            response = input("✅ Répondez (o/n): ").strip().lower()
            
            if response in ['o', 'oui', 'y', 'yes']:
                print("🎊 VALIDATION HUMAINE RÉUSSIE!")
                
                # Sauvegarde configuration validée
                config_file = Path("PIPELINE/config/tts_phase3_validated.yaml")
                config_file.parent.mkdir(exist_ok=True)
                
                import yaml
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                print(f"💾 Configuration sauvegardée: {config_file}")
                
                # Métriques finales
                metrics = {
                    "validation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "backend_used": result.backend_used,
                    "latency_ms": latency_ms,
                    "audio_size_bytes": len(result.audio_data),
                    "gpu_config": "RTX 3090 (CUDA:1)",
                    "human_validation": "SUCCESS",
                    "sapi_used": False,
                    "phase3_config": True
                }
                
                metrics_file = Path("PIPELINE/reports/tts_phase3_validation.json")
                metrics_file.parent.mkdir(exist_ok=True)
                
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                
                print(f"📊 Métriques sauvegardées: {metrics_file}")
                return True
            else:
                print("❌ VALIDATION HUMAINE ÉCHOUÉE")
                return False
                
        else:
            print("🚫 ÉCHEC TTS: Aucun audio généré")
            if result:
                print(f"❌ Erreur: {result.error}")
            return False
            
    except Exception as e:
        print(f"💥 ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Point d'entrée principal"""
    print("🎯 TEST TTS PHASE 3 VALIDÉE - CONFIGURATION SANS SAPI")
    print("🚨 RTX 3090 (CUDA:1) OBLIGATOIRE")
    print("📋 Backends: PiperNative + PiperCLI + SilentEmergency")
    print()
    
    success = await test_tts_phase3_validated()
    
    if success:
        print("\n🎊 SUCCÈS COMPLET - TTS PHASE 3 VALIDÉ!")
        print("✅ Configuration TTS fonctionnelle sans SAPI")
        print("✅ Validation humaine confirmée")
        print("✅ Prêt pour intégration pipeline")
    else:
        print("\n❌ ÉCHEC - TTS Phase 3 non validé")
        print("🔧 Vérifiez la configuration et les backends")
    
    return success

if __name__ == "__main__":
    asyncio.run(main()) 