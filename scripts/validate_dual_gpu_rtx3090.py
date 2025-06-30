#!/usr/bin/env python3
"""
Validation configuration RTX 3090 SuperWhisper V6 - Phase 4 STT
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Script de validation obligatoire avant toute implémentation STT

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

def validate_environment():
    """Validation environnement Python et dépendances"""
    print("🔍 Validation environnement Python...")
    
    # Version Python
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 9:
        print(f"❌ Python {python_version.major}.{python_version.minor} - Version 3.9+ requise")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Dépendances critiques
    required_packages = ['torch', 'numpy', 'asyncio']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} disponible")
        except ImportError:
            print(f"❌ {package} manquant")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Packages manquants: {', '.join(missing_packages)}")
        return False
    
    return True

def validate_cuda_configuration():
    """Validation configuration CUDA"""
    print("\n🔍 Validation configuration CUDA...")
    
    # Variables environnement
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER', '')
    pytorch_alloc = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    
    print(f"🔒 CUDA_VISIBLE_DEVICES: '{cuda_devices}'")
    print(f"🔒 CUDA_DEVICE_ORDER: '{cuda_order}'")
    print(f"🔒 PYTORCH_CUDA_ALLOC_CONF: '{pytorch_alloc}'")
    
    # Validation configuration
    if cuda_devices != '1':
        print(f"❌ CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        return False
    
    if cuda_order != 'PCI_BUS_ID':
        print(f"❌ CUDA_DEVICE_ORDER='{cuda_order}' incorrect - doit être 'PCI_BUS_ID'")
        return False
    
    print("✅ Configuration CUDA correcte")
    return True

def validate_gpu_hardware():
    """Validation matériel GPU RTX 3090"""
    print("\n🔍 Validation matériel GPU...")
    
    try:
        import torch
    except ImportError:
        print("❌ PyTorch non installé")
        return False
    
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible")
        return False
    
    # Informations GPU
    device_count = torch.cuda.device_count()
    print(f"🎮 Devices CUDA détectés: {device_count}")
    
    if device_count == 0:
        print("❌ Aucun device CUDA disponible")
        return False
    
    # GPU principal (cuda:0 après mapping)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🎮 GPU 0 (mappé): {gpu_name}")
    print(f"💾 VRAM: {gpu_memory:.1f}GB")
    
    # Validation RTX 3090
    if "RTX 3090" not in gpu_name:
        print(f"❌ GPU détecté: {gpu_name}")
        print("❌ RTX 3090 requise pour SuperWhisper V6 STT")
        return False
    
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        print(f"❌ VRAM {gpu_memory:.1f}GB insuffisante")
        print("❌ RTX 3090 24GB requise")
        return False
    
    print("✅ RTX 3090 validée:")
    print(f"   Nom: {gpu_name}")
    print(f"   VRAM: {gpu_memory:.1f}GB")
    print("   Configuration: STT + TTS séquentiel sur même GPU")
    
    return True

def test_gpu_allocation():
    """Test allocation mémoire GPU"""
    print("\n🔍 Test allocation mémoire GPU...")
    
    try:
        import torch
        
        # Test allocation basique
        device = torch.device("cuda:0")
        test_tensor = torch.randn(1000, 1000, device=device)
        
        # Vérification allocation
        allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        print(f"✅ Test allocation réussi: {allocated:.1f}MB alloués")
        
        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur allocation GPU: {e}")
        return False

def validate_stt_readiness():
    """Validation préparation STT"""
    print("\n🔍 Validation préparation STT...")
    
    # Structure dossiers
    required_dirs = [
        'STT',
        'STT/backends',
        'STT/utils',
        'STT/config',
        'tests/STT',
        'scripts/STT'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"✅ Dossier: {dir_path}")
    
    if missing_dirs:
        print(f"❌ Dossiers manquants: {', '.join(missing_dirs)}")
        return False
    
    print("✅ Structure STT prête")
    return True

def main():
    """Validation complète configuration RTX 3090"""
    print("🚀 VALIDATION CONFIGURATION RTX 3090 - SUPERWHISPER V6 PHASE 4 STT")
    print("=" * 70)
    
    validations = [
        ("Environnement Python", validate_environment),
        ("Configuration CUDA", validate_cuda_configuration),
        ("Matériel GPU RTX 3090", validate_gpu_hardware),
        ("Allocation GPU", test_gpu_allocation),
        ("Préparation STT", validate_stt_readiness)
    ]
    
    all_passed = True
    
    for name, validation_func in validations:
        try:
            if not validation_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Erreur validation {name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎊 VALIDATION RÉUSSIE - RTX 3090 PRÊTE POUR STT")
        print("✅ Configuration SuperWhisper V6 Phase 4 STT validée")
        print("🚀 Prêt pour implémentation Prism_Whisper2")
    else:
        print("❌ VALIDATION ÉCHOUÉE - CONFIGURATION INCORRECTE")
        print("🔧 Corrigez les erreurs avant de continuer")
        sys.exit(1)

if __name__ == "__main__":
    main() 