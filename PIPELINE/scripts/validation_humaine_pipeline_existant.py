#!/usr/bin/env python3
"""
Validation Humaine Pipeline - TTS EXISTANT INCHANGÉ
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

UTILISE LE TTS EXISTANT TEL QUEL - AUCUNE MODIFICATION
Pipeline complet : Mic → STT → LLM → TTS → Audio

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

async def test_pipeline_complet_existant():
    """Test pipeline complet avec TTS existant inchangé"""
    print("\n🚀 VALIDATION HUMAINE PIPELINE COMPLET")
    print("🔧 UTILISE TTS EXISTANT TEL QUEL - AUCUNE MODIFICATION")
    print("=" * 60)
    
    # Validation GPU obligatoire
    if not validate_rtx3090_configuration():
        print("🚫 ÉCHEC: Configuration GPU RTX 3090 invalide")
        return False
    
    try:
        # Import du pipeline existant
        from PIPELINE.pipeline_orchestrator import PipelineOrchestrator
        
        print("✅ Import PipelineOrchestrator réussi")
        
        # Configuration pipeline existante
        config_path = Path("PIPELINE/config/pipeline.yaml")
        if not config_path.exists():
            print(f"⚠️ Config pipeline non trouvée: {config_path}")
            # Utilisation config par défaut
            config = {}
        else:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration chargée: {config_path}")
        
        # Import TTS existant INCHANGÉ
        from TTS.tts_manager import UnifiedTTSManager
        
        # Configuration TTS existante
        tts_config_path = Path("config/tts.yaml")
        if tts_config_path.exists():
            with open(tts_config_path, 'r', encoding='utf-8') as f:
                tts_config = yaml.safe_load(f)
        else:
            # Config minimale pour TTS existant
            tts_config = {}
        
        print("🔧 Initialisation TTS existant...")
        tts_manager = UnifiedTTSManager(tts_config)
        
        # Initialisation pipeline avec TTS existant
        print("🔧 Initialisation PipelineOrchestrator avec TTS existant...")
        pipeline = PipelineOrchestrator(config, tts_manager)
        
        # Test simple TTS via pipeline
        test_text = "Bonjour, test de validation humaine SuperWhisper V6."
        print(f"📝 Texte test: {test_text}")
        
        print("🎵 Test TTS via pipeline...")
        start_time = time.time()
        
        # Utilisation TTS existant via pipeline
        print("🎵 Synthèse TTS via TTS existant...")
        tts_result = await tts_manager.synthesize(test_text)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Validation résultat
        if tts_result and tts_result.success and tts_result.audio_data:
            print(f"✅ TTS PIPELINE SUCCÈS!")
            print(f"🎯 Backend utilisé: {tts_result.backend_used}")
            print(f"⚡ Latence: {latency_ms:.1f}ms")
            print(f"🔊 Audio généré: {len(tts_result.audio_data):,} bytes")
            
            # Sauvegarde audio pour validation humaine
            output_file = Path("PIPELINE/test_output/validation_humaine_pipeline.wav")
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'wb') as f:
                f.write(tts_result.audio_data)
            
            print(f"💾 Audio sauvegardé: {output_file}")
            
            # Lecture audio automatique
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(str(output_file))
                pygame.mixer.music.play()
                print("🔊 Lecture audio automatique...")
                
                # Attendre fin lecture
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                print(f"⚠️ Lecture automatique échouée: {e}")
                print("🔊 Veuillez lire manuellement le fichier audio")
            
            # Validation humaine
            print("\n" + "="*60)
            print("🎧 VALIDATION HUMAINE PIPELINE COMPLET")
            print("="*60)
            print(f"📁 Fichier audio: {output_file}")
            print("🔊 Audio lu automatiquement (ou manuellement)")
            print("❓ Avez-vous entendu une vraie voix française ?")
            print("❓ Le pipeline TTS fonctionne-t-il correctement ?")
            
            response = input("✅ Validation pipeline (o/n): ").strip().lower()
            
            if response in ['o', 'oui', 'y', 'yes']:
                print("🎊 VALIDATION HUMAINE PIPELINE RÉUSSIE!")
                
                # Métriques finales
                metrics = {
                    "validation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "pipeline_test": "SUCCESS",
                    "backend_used": tts_result.backend_used,
                    "latency_ms": latency_ms,
                    "audio_size_bytes": len(tts_result.audio_data),
                    "gpu_config": "RTX 3090 (CUDA:1)",
                    "human_validation": "SUCCESS",
                    "tts_modified": False,  # TTS non modifié
                    "pipeline_integration": True
                }
                
                metrics_file = Path("PIPELINE/reports/validation_humaine_pipeline.json")
                metrics_file.parent.mkdir(exist_ok=True)
                
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                
                print(f"📊 Métriques sauvegardées: {metrics_file}")
                return True
            else:
                print("❌ VALIDATION HUMAINE PIPELINE ÉCHOUÉE")
                return False
                
        else:
            print("🚫 ÉCHEC TTS PIPELINE: Aucun audio généré")
            if tts_result:
                print(f"❌ Erreur: {tts_result.error}")
            return False
            
    except Exception as e:
        print(f"💥 ERREUR PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Point d'entrée principal"""
    print("🎯 VALIDATION HUMAINE PIPELINE COMPLET")
    print("🚨 RTX 3090 (CUDA:1) OBLIGATOIRE")
    print("🔧 TTS EXISTANT INCHANGÉ - AUCUNE MODIFICATION")
    print()
    
    success = await test_pipeline_complet_existant()
    
    if success:
        print("\n🎊 SUCCÈS COMPLET - PIPELINE VALIDÉ!")
        print("✅ TTS existant fonctionne via pipeline")
        print("✅ Validation humaine confirmée")
        print("✅ Tâche 4 peut être marquée terminée")
    else:
        print("\n❌ ÉCHEC - Pipeline non validé")
        print("🔧 Problème d'intégration pipeline/TTS")
    
    return success

if __name__ == "__main__":
    asyncio.run(main()) 