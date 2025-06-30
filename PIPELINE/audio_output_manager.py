#!/usr/bin/env python3
"""
AudioOutputManager - Gestionnaire de sortie audio
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Module simple pour lecture audio dans le pipeline complet

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
import tempfile
import subprocess
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU

print("🎮 AudioOutputManager - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

class AudioOutputManager:
    """Gestionnaire simple de sortie audio"""
    
    def __init__(self):
        """Initialisation du gestionnaire audio"""
        self.temp_dir = Path(tempfile.gettempdir()) / "superwhisper_audio"
        self.temp_dir.mkdir(exist_ok=True)
        print("✅ AudioOutputManager initialisé")
    
    async def play_audio(self, audio_data: bytes) -> bool:
        """
        Lecture audio simple
        
        Args:
            audio_data: Données audio en bytes
            
        Returns:
            bool: True si succès
        """
        try:
            # Sauvegarde temporaire
            temp_file = self.temp_dir / f"temp_audio_{os.getpid()}.wav"
            
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            print(f"🔊 Lecture audio: {len(audio_data):,} bytes")
            
            # Lecture avec Windows Media Player ou autre
            if os.name == 'nt':  # Windows
                # Utilisation de PowerShell pour lecture audio
                cmd = [
                    "powershell", "-Command",
                    f"(New-Object Media.SoundPlayer '{temp_file}').PlaySync()"
                ]
                
                # Exécution asynchrone
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    print("✅ Audio lu avec succès")
                    return True
                else:
                    print(f"⚠️ Erreur lecture audio: {stderr.decode()}")
                    return False
            else:
                # Linux/Mac - utiliser aplay ou afplay
                print("⚠️ Lecture audio non implémentée pour ce système")
                return False
                
        except Exception as e:
            print(f"❌ Erreur AudioOutputManager: {e}")
            return False
        finally:
            # Nettoyage
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
    
    def close(self):
        """Fermeture propre du gestionnaire"""
        try:
            # Nettoyage répertoire temporaire
            if self.temp_dir.exists():
                for file in self.temp_dir.glob("temp_audio_*.wav"):
                    try:
                        file.unlink()
                    except:
                        pass
            print("✅ AudioOutputManager fermé")
        except Exception as e:
            print(f"⚠️ Erreur fermeture AudioOutputManager: {e}")

# Test simple
if __name__ == "__main__":
    async def test_audio():
        manager = AudioOutputManager()
        
        # Test avec données vides
        success = await manager.play_audio(b"test")
        print(f"Test audio: {'✅ Succès' if success else '❌ Échec'}")
        
        manager.close()
    
    asyncio.run(test_audio()) 