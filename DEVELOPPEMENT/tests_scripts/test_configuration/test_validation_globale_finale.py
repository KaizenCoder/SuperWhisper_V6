#!/usr/bin/env python3
"""
VALIDATION GLOBALE FINALE - TOUTES CORRECTIONS GPU
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:0) EXCLUSIVE

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

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # RTX 3090 EXCLUSIVEMENT
print("🎮 VALIDATION GLOBALE: RTX 3090 (CUDA:0) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def validate_all_corrections():
    """Validation globale de toutes les corrections GPU"""
    print("\n🔍 VALIDATION GLOBALE - TOUTES CORRECTIONS GPU")
    print("="*60)
    
    all_validations = []
    
    # FICHIERS CORRIGÉS À VÉRIFIER
    files_to_check = [
        {
            'file': 'tests/test_stt_handler.py',
            'checks': [
                ('cuda:0', "Configuration cuda:0 RTX 3090"),
                ('RTX 3090 (CUDA:0)', "Documentation RTX 3090")
            ]
        },
        {
            'file': 'utils/gpu_manager.py',
            'checks': [
                ('cuda:0', "Configuration cuda:0 RTX 3090"),
                ('RTX 3090 (CUDA:0)', "Documentation RTX 3090")
            ]
        },
        {
            'file': 'tests/test_llm_handler.py',
            'checks': [
                ("'gpu_device_index': 0", "Index GPU 0 RTX 3090"),
                ('RTX 3090 (CUDA:0)', "Documentation RTX 3090")
            ]
        },
        {
            'file': 'STT/vad_manager.py',
            'checks': [
                ('cuda:0', "Configuration cuda:0 RTX 3090"),
                ('torch.cuda.set_device(0)', "Set device 0 RTX 3090")
            ]
        }
    ]
    
    # VÉRIFICATION DE CHAQUE FICHIER
    for file_info in files_to_check:
        filename = file_info['file']
        print(f"\n📁 FICHIER: {filename}")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_valid = True
            for check_pattern, description in file_info['checks']:
                if check_pattern in content:
                    print(f"   ✅ {description}")
                    all_validations.append(f"✅ {filename}: {description}")
                else:
                    print(f"   ❌ MANQUANT: {description}")
                    all_validations.append(f"❌ {filename}: MANQUANT {description}")
                    file_valid = False
            
            # Vérification anti-pattern (pas de cuda:1 non documenté)
            if 'cuda:1' in content and 'RTX 5060' not in content:
                print(f"   ❌ ERREUR: cuda:1 trouvé sans doc RTX 5060")
                all_validations.append(f"❌ {filename}: cuda:1 non documenté")
                file_valid = False
            else:
                print(f"   ✅ Aucun cuda:1 non documenté")
                all_validations.append(f"✅ {filename}: Aucun cuda:1 non documenté")
                
        except FileNotFoundError:
            print(f"   ❌ FICHIER NON TROUVÉ: {filename}")
            all_validations.append(f"❌ {filename}: FICHIER NON TROUVÉ")
        except Exception as e:
            print(f"   ❌ ERREUR LECTURE: {e}")
            all_validations.append(f"❌ {filename}: ERREUR {e}")
    
    # VALIDATION GPU MATÉRIELLE
    print(f"\n🎮 VALIDATION GPU MATÉRIELLE")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"   ✅ GPU détectée: {gpu_name}")
        print(f"   ✅ VRAM disponible: {gpu_memory:.1f}GB")
        
        if "RTX 3090" in gpu_name:
            print(f"   ✅ RTX 3090 confirmée")
            all_validations.append("✅ HARDWARE: RTX 3090 confirmée")
        else:
            print(f"   ❌ GPU incorrecte: {gpu_name}")
            all_validations.append(f"❌ HARDWARE: GPU incorrecte {gpu_name}")
            
        if gpu_memory >= 20:
            print(f"   ✅ VRAM suffisante (≥20GB)")
            all_validations.append("✅ HARDWARE: VRAM suffisante")
        else:
            print(f"   ❌ VRAM insuffisante: {gpu_memory}GB")
            all_validations.append(f"❌ HARDWARE: VRAM insuffisante {gpu_memory}GB")
    else:
        print("   ⚠️ CUDA non disponible - validation matérielle impossible")
        all_validations.append("⚠️ HARDWARE: CUDA non disponible")
    
    # RAPPORT FINAL
    print(f"\n📊 RAPPORT FINAL VALIDATION GLOBALE")
    print("="*60)
    
    successes = [v for v in all_validations if v.startswith('✅')]
    failures = [v for v in all_validations if v.startswith('❌')]
    warnings = [v for v in all_validations if v.startswith('⚠️')]
    
    print(f"✅ SUCCÈS: {len(successes)}")
    print(f"❌ ÉCHECS: {len(failures)}")
    print(f"⚠️ AVERTISSEMENTS: {len(warnings)}")
    
    if failures:
        print(f"\n🚫 ÉCHECS DÉTECTÉS:")
        for failure in failures:
            print(f"   {failure}")
        return False
    else:
        print(f"\n🎉 TOUTES VALIDATIONS RÉUSSIES!")
        print("   🔒 Configuration RTX 3090 (CUDA:0) exclusive confirmée")
        print("   🚫 Aucune référence RTX 5060 (CUDA:1) non documentée")
        print("   ✅ Tous les fichiers critiques corrigés")
        return True

if __name__ == "__main__":
    success = validate_all_corrections()
    if success:
        print("\n🎉 MISSION CORRECTION GPU ACCOMPLIE AVEC SUCCÈS!")
        print("🔒 PROJET SUPERWHISPER V6 SÉCURISÉ RTX 3090 EXCLUSIVE")
        sys.exit(0)
    else:
        print("\n🚫 MISSION CORRECTION GPU ÉCHOUÉE - VIOLATIONS DÉTECTÉES")
        sys.exit(1) 