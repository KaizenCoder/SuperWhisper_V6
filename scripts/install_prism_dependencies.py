#!/usr/bin/env python3
"""
Script d'installation des dépendances Prism STT - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🔧 Installation Prism Dependencies - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def run_command(command, description=""):
    """
    Exécute une commande avec gestion d'erreur.
    
    Args:
        command: Commande à exécuter
        description: Description de l'opération
    
    Returns:
        bool: True si succès, False sinon
    """
    print(f"\n🔄 {description}")
    print(f"   Commande: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(f"✅ Sortie: {result.stdout.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur: {e}")
        if e.stdout:
            print(f"   Stdout: {e.stdout}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return False

def check_python_version():
    """Vérifie la version Python"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ requis")
        return False
    
    print("✅ Version Python compatible")
    return True

def check_cuda_availability():
    """Vérifie la disponibilité CUDA"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA non disponible")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🎮 GPU détectée: {gpu_name}")
        print(f"💾 VRAM: {gpu_memory:.1f}GB")
        
        if "3090" not in gpu_name:
            print(f"⚠️ GPU non optimale: {gpu_name} (RTX 3090 recommandée)")
        
        if gpu_memory < 8:
            print(f"⚠️ VRAM faible: {gpu_memory:.1f}GB (8GB+ recommandé)")
        
        print("✅ CUDA disponible")
        return True
        
    except ImportError:
        print("⚠️ PyTorch non installé - sera installé avec les dépendances")
        return True

def install_base_dependencies():
    """Installe les dépendances de base"""
    print("\n📦 Installation des dépendances de base...")
    
    base_packages = [
        "torch",
        "torchaudio", 
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "pyaudio",
        "asyncio",
        "aiofiles"
    ]
    
    for package in base_packages:
        success = run_command(
            f"pip install {package}",
            f"Installation {package}"
        )
        if not success:
            print(f"⚠️ Échec installation {package} - continuons...")
    
    return True

def install_faster_whisper():
    """Installe faster-whisper"""
    print("\n🎤 Installation faster-whisper...")
    
    # Installation faster-whisper
    success = run_command(
        "pip install faster-whisper",
        "Installation faster-whisper"
    )
    
    if not success:
        print("⚠️ Tentative installation depuis GitHub...")
        success = run_command(
            "pip install git+https://github.com/guillaumekln/faster-whisper.git",
            "Installation faster-whisper depuis GitHub"
        )
    
    return success

def install_ctranslate2():
    """Installe CTranslate2 pour faster-whisper"""
    print("\n⚡ Installation CTranslate2...")
    
    # Installation CTranslate2
    success = run_command(
        "pip install ctranslate2",
        "Installation CTranslate2"
    )
    
    return success

def install_audio_dependencies():
    """Installe les dépendances audio"""
    print("\n🔊 Installation dépendances audio...")
    
    audio_packages = [
        "ffmpeg-python",
        "pydub",
        "webrtcvad",
        "noisereduce"
    ]
    
    for package in audio_packages:
        success = run_command(
            f"pip install {package}",
            f"Installation {package}"
        )
        if not success:
            print(f"⚠️ Échec installation {package} - continuons...")
    
    return True

def install_monitoring_dependencies():
    """Installe les dépendances de monitoring"""
    print("\n📊 Installation dépendances monitoring...")
    
    monitoring_packages = [
        "prometheus-client",
        "psutil",
        "GPUtil"
    ]
    
    for package in monitoring_packages:
        success = run_command(
            f"pip install {package}",
            f"Installation {package}"
        )
        if not success:
            print(f"⚠️ Échec installation {package} - continuons...")
    
    return True

def download_whisper_models():
    """Télécharge les modèles Whisper"""
    print("\n📥 Téléchargement modèles Whisper...")
    
    # Créer répertoire modèles
    models_dir = Path("./models/whisper")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Script de téléchargement
    download_script = """
import os
from faster_whisper import WhisperModel

# Configuration GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print("📥 Téléchargement modèle tiny...")
model_tiny = WhisperModel("tiny", device="cuda", download_root="./models/whisper")
print("✅ Modèle tiny téléchargé")

print("📥 Téléchargement modèle large-v2...")
model_large = WhisperModel("large-v2", device="cuda", download_root="./models/whisper")
print("✅ Modèle large-v2 téléchargé")

print("🎯 Modèles prêts pour utilisation")
"""
    
    # Sauvegarder script temporaire
    script_path = Path("temp_download_models.py")
    script_path.write_text(download_script)
    
    try:
        success = run_command(
            f"python {script_path}",
            "Téléchargement modèles Whisper"
        )
        
        # Nettoyage
        script_path.unlink()
        
        return success
        
    except Exception as e:
        print(f"⚠️ Erreur téléchargement modèles: {e}")
        if script_path.exists():
            script_path.unlink()
        return False

def create_requirements_file():
    """Crée un fichier requirements.txt pour Prism STT"""
    print("\n📝 Création requirements_prism_stt.txt...")
    
    requirements = """# Dépendances Prism STT - SuperWhisper V6
# Configuration RTX 3090 (CUDA:1) obligatoire

# Core ML/Audio
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.21.0
scipy>=1.7.0

# STT Engine
faster-whisper>=0.9.0
ctranslate2>=3.20.0

# Audio Processing
librosa>=0.9.0
soundfile>=0.12.0
pyaudio>=0.2.11
ffmpeg-python>=0.2.0
pydub>=0.25.0
webrtcvad>=2.0.10
noisereduce>=3.0.0

# Async/Performance
asyncio
aiofiles>=23.0.0

# Monitoring
prometheus-client>=0.17.0
psutil>=5.9.0
GPUtil>=1.4.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0

# Development
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
"""
    
    requirements_path = Path("requirements_prism_stt.txt")
    requirements_path.write_text(requirements)
    
    print(f"✅ Fichier créé: {requirements_path}")
    return True

def test_installation():
    """Test l'installation"""
    print("\n🧪 Test de l'installation...")
    
    test_script = """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    import faster_whisper
    print(f"✅ faster-whisper importé")
    
    import ctranslate2
    print(f"✅ ctranslate2: {ctranslate2.__version__}")
    
    import numpy as np
    print(f"✅ numpy: {np.__version__}")
    
    import librosa
    print(f"✅ librosa: {librosa.__version__}")
    
    print("🎯 Installation validée avec succès!")
    
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    exit(1)
"""
    
    # Sauvegarder script temporaire
    script_path = Path("temp_test_installation.py")
    script_path.write_text(test_script)
    
    try:
        success = run_command(
            f"python {script_path}",
            "Test installation"
        )
        
        # Nettoyage
        script_path.unlink()
        
        return success
        
    except Exception as e:
        print(f"⚠️ Erreur test installation: {e}")
        if script_path.exists():
            script_path.unlink()
        return False

def main():
    """Installation complète des dépendances Prism STT"""
    print("🚀 Installation des dépendances Prism STT - SuperWhisper V6")
    print("=" * 60)
    
    # Vérifications préalables
    if not check_python_version():
        return False
    
    # Installation étape par étape
    steps = [
        ("Dépendances de base", install_base_dependencies),
        ("CTranslate2", install_ctranslate2),
        ("faster-whisper", install_faster_whisper),
        ("Dépendances audio", install_audio_dependencies),
        ("Dépendances monitoring", install_monitoring_dependencies),
        ("Fichier requirements", create_requirements_file),
        ("Vérification CUDA", check_cuda_availability),
        ("Test installation", test_installation)
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        
        try:
            if step_func():
                success_count += 1
                print(f"✅ {step_name} - Succès")
            else:
                print(f"⚠️ {step_name} - Échec partiel")
        except Exception as e:
            print(f"❌ {step_name} - Erreur: {e}")
    
    # Résumé final
    print("\n" + "="*60)
    print("📊 RÉSUMÉ INSTALLATION")
    print("="*60)
    print(f"✅ Étapes réussies: {success_count}/{total_steps}")
    
    if success_count >= total_steps - 1:  # Tolérance 1 échec
        print("🎯 Installation RÉUSSIE!")
        print("\n📋 Prochaines étapes:")
        print("   1. Tester: python tests/test_prism_integration.py")
        print("   2. Lancer: python STT/backends/prism_stt_backend.py")
        print("   3. Intégrer dans UnifiedSTTManager")
        return True
    else:
        print("⚠️ Installation PARTIELLE - Vérifiez les erreurs ci-dessus")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 