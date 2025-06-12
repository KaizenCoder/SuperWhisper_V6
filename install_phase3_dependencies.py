#!/usr/bin/env python3
"""
Installation des Dépendances Phase 3 - SuperWhisper V6 TTS
Installation automatique du binding Python Piper et autres optimisations
🚀 Prérequis pour les optimisations de performance
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Phase3DependencyInstaller:
    """
    Installateur des dépendances Phase 3
    
    🚀 COMPOSANTS INSTALLÉS:
    1. Binding Python Piper (piper-tts)
    2. Dépendances audio (wave, asyncio)
    3. Outils de performance (psutil, memory_profiler)
    4. Validation de l'environnement GPU
    """
    
    def __init__(self):
        self.python_executable = sys.executable
        self.installation_log = []
        
        print("🔧 Phase 3 Dependency Installer initialisé")
        print(f"🐍 Python: {self.python_executable}")
    
    def run_installation(self):
        """Exécution complète de l'installation"""
        print("\n" + "="*80)
        print("🚀 INSTALLATION DÉPENDANCES PHASE 3")
        print("="*80)
        
        try:
            # 1. Vérification de l'environnement
            self._check_environment()
            
            # 2. Installation du binding Python Piper
            self._install_piper_binding()
            
            # 3. Installation des dépendances audio
            self._install_audio_dependencies()
            
            # 4. Installation des outils de performance
            self._install_performance_tools()
            
            # 5. Validation finale
            self._validate_installation()
            
            # 6. Rapport final
            self._generate_report()
            
        except Exception as e:
            print(f"❌ Erreur installation: {e}")
            self._generate_error_report(e)
    
    def _check_environment(self):
        """Vérification de l'environnement système"""
        print("\n🔍 VÉRIFICATION ENVIRONNEMENT")
        print("-" * 50)
        
        # Version Python
        python_version = sys.version_info
        print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            raise RuntimeError("Python 3.8+ requis pour Phase 3")
        
        # Vérification pip
        try:
            import pip
            print(f"📦 pip: {pip.__version__}")
        except ImportError:
            raise RuntimeError("pip non disponible")
        
        # Vérification GPU (optionnelle)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🎮 GPU: {gpu_name}")
            else:
                print("⚠️ CUDA non disponible (mode CPU)")
        except ImportError:
            print("⚠️ PyTorch non installé (GPU optionnel)")
        
        print("✅ Environnement validé")
    
    def _install_piper_binding(self):
        """Installation du binding Python Piper"""
        print("\n🔧 INSTALLATION BINDING PYTHON PIPER")
        print("-" * 50)
        
        # Tentative d'installation via pip
        piper_packages = [
            'piper-tts',           # Package principal
            'piper-phonemize',     # Phonémisation
            'onnxruntime-gpu',     # Runtime ONNX GPU
        ]
        
        for package in piper_packages:
            try:
                print(f"📦 Installation {package}...")
                result = subprocess.run(
                    [self.python_executable, '-m', 'pip', 'install', package],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✅ {package} installé avec succès")
                self.installation_log.append(f"SUCCESS: {package}")
                
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Échec installation {package}: {e}")
                print(f"   Sortie: {e.stdout}")
                print(f"   Erreur: {e.stderr}")
                self.installation_log.append(f"FAILED: {package} - {e}")
                
                # Tentative alternative pour piper-tts
                if package == 'piper-tts':
                    print("🔄 Tentative installation alternative...")
                    self._install_piper_alternative()
    
    def _install_piper_alternative(self):
        """Installation alternative de Piper (compilation depuis source)"""
        print("🛠️ Installation alternative Piper depuis source...")
        
        # Instructions pour installation manuelle
        print("""
📋 INSTALLATION MANUELLE PIPER:

1. Télécharger les binaires Piper:
   https://github.com/rhasspy/piper/releases

2. Extraire dans le dossier piper/ du projet

3. Installer les dépendances Python:
   pip install onnxruntime-gpu numpy

4. Tester avec:
   python test_phase3_optimisations.py

⚠️ Le binding Python natif est optionnel.
   Le système utilisera le fallback CLI si indisponible.
        """)
        
        self.installation_log.append("INFO: Instructions manuelles Piper fournies")
    
    def _install_audio_dependencies(self):
        """Installation des dépendances audio"""
        print("\n🎵 INSTALLATION DÉPENDANCES AUDIO")
        print("-" * 50)
        
        audio_packages = [
            'wave',                # Manipulation WAV (built-in)
            'pydub',              # Manipulation audio avancée
            'soundfile',          # Lecture/écriture audio
        ]
        
        for package in audio_packages:
            if package == 'wave':
                print(f"✅ {package} (built-in Python)")
                continue
                
            try:
                print(f"📦 Installation {package}...")
                subprocess.run(
                    [self.python_executable, '-m', 'pip', 'install', package],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✅ {package} installé")
                self.installation_log.append(f"SUCCESS: {package}")
                
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Échec {package}: {e}")
                self.installation_log.append(f"FAILED: {package}")
    
    def _install_performance_tools(self):
        """Installation des outils de performance"""
        print("\n⚡ INSTALLATION OUTILS PERFORMANCE")
        print("-" * 50)
        
        perf_packages = [
            'psutil',             # Monitoring système
            'memory-profiler',    # Profiling mémoire
            'pyyaml',            # Configuration YAML
        ]
        
        for package in perf_packages:
            try:
                print(f"📦 Installation {package}...")
                subprocess.run(
                    [self.python_executable, '-m', 'pip', 'install', package],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✅ {package} installé")
                self.installation_log.append(f"SUCCESS: {package}")
                
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Échec {package}: {e}")
                self.installation_log.append(f"FAILED: {package}")
    
    def _validate_installation(self):
        """Validation de l'installation"""
        print("\n✅ VALIDATION INSTALLATION")
        print("-" * 50)
        
        # Test des imports critiques
        test_imports = [
            ('yaml', 'Configuration YAML'),
            ('asyncio', 'Programmation asynchrone'),
            ('threading', 'Threading (built-in)'),
            ('collections', 'Collections (built-in)'),
            ('hashlib', 'Hashing (built-in)'),
            ('wave', 'Audio WAV (built-in)'),
        ]
        
        for module, description in test_imports:
            try:
                __import__(module)
                print(f"✅ {module}: {description}")
            except ImportError:
                print(f"❌ {module}: {description} - MANQUANT")
        
        # Test optionnel du binding Piper
        try:
            import piper
            print("✅ piper: Binding Python natif disponible")
        except ImportError:
            print("⚠️ piper: Binding Python non disponible (fallback CLI sera utilisé)")
        
        # Test des composants Phase 3
        try:
            sys.path.insert(0, str(Path.cwd()))
            from TTS.utils_audio import is_valid_wav
            print("✅ TTS.utils_audio: Utilitaires audio disponibles")
        except ImportError as e:
            print(f"⚠️ TTS.utils_audio: {e}")
    
    def _generate_report(self):
        """Génération du rapport d'installation"""
        print("\n" + "="*80)
        print("📊 RAPPORT INSTALLATION PHASE 3")
        print("="*80)
        
        # Comptage des succès/échecs
        successes = [log for log in self.installation_log if log.startswith('SUCCESS')]
        failures = [log for log in self.installation_log if log.startswith('FAILED')]
        
        print(f"✅ Succès: {len(successes)}")
        print(f"❌ Échecs: {len(failures)}")
        print()
        
        if failures:
            print("⚠️ ÉCHECS D'INSTALLATION:")
            for failure in failures:
                print(f"   {failure}")
            print()
        
        # Instructions post-installation
        print("🚀 PROCHAINES ÉTAPES:")
        print("1. Exécuter: python test_phase3_optimisations.py")
        print("2. Vérifier les performances avec les nouveaux handlers")
        print("3. Activer piper_native_optimized dans config/tts.yaml")
        print("4. Tester avec des textes longs (5000+ caractères)")
        print()
        
        # Statut global
        if len(failures) == 0:
            print("🎉 Installation Phase 3 COMPLÈTE!")
        elif len(failures) <= 2:
            print("⚠️ Installation Phase 3 PARTIELLE (fonctionnalités limitées)")
        else:
            print("❌ Installation Phase 3 ÉCHOUÉE (révision requise)")
    
    def _generate_error_report(self, error):
        """Génération du rapport d'erreur"""
        print("\n" + "="*80)
        print("❌ RAPPORT D'ERREUR INSTALLATION")
        print("="*80)
        
        print(f"Erreur: {error}")
        print()
        print("🔧 SOLUTIONS POSSIBLES:")
        print("1. Vérifier la connexion Internet")
        print("2. Mettre à jour pip: python -m pip install --upgrade pip")
        print("3. Installer manuellement: pip install piper-tts")
        print("4. Utiliser un environnement virtuel")
        print("5. Vérifier les permissions d'écriture")
        print()
        print("📞 Support: Consulter la documentation SuperWhisper V6")


def main():
    """Point d'entrée principal"""
    installer = Phase3DependencyInstaller()
    installer.run_installation()


if __name__ == "__main__":
    main() 