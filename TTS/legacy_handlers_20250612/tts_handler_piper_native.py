#!/usr/bin/env python3
"""
TTS/tts_handler_piper_native.py
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Handler TTS Piper Native avec configuration GPU RTX 3090 exclusive

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

import subprocess
import tempfile
import wave
import numpy as np
import sounddevice as sd

def validate_rtx3090_mandatory():
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
    except ImportError:
        print("⚠️ PyTorch non disponible - validation GPU ignorée pour TTS Piper Native")

class TTSHandlerPiperNative:
    def __init__(self, config):
        # Validation RTX 3090 obligatoire
        validate_rtx3090_mandatory()
        
        self.config = config
        self.model_path = config.get('model_path', 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx')
        self.sample_rate = config.get('sample_rate', 22050)
        
        print(f"🎮 TTS Handler Piper Native RTX 3090 initialisé")
        print(f"📦 Modèle: {self.model_path}")
        
        # Vérifier que le modèle existe
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        
        # Chercher l'executable piper
        self._find_piper_executable()
        
    def _find_piper_executable(self):
        """Trouver l'executable piper"""
        # Chercher dans le venv actuel
        possible_paths = [
            os.path.join(os.getcwd(), 'venv_piper312', 'Scripts', 'piper.exe'),
            os.path.join(os.getcwd(), 'venv_piper312', 'bin', 'piper'),
            'piper.exe',
            'piper'
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, '--help'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self.piper_path = path
                    print(f"✅ Piper trouvé: {path}")
                    return
            except:
                continue
                
        # Fallback: essayer d'utiliser notre piper compilé
        piper_dir = os.path.join(os.getcwd(), 'venv_piper312', 'Lib', 'site-packages', 'piper_tts')
        if os.path.exists(piper_dir):
            print(f"📁 Répertoire piper trouvé: {piper_dir}")
            
        raise RuntimeError("Executable piper non trouvé")
        
    def synthesize_with_cli(self, text, output_path):
        """Synthèse avec CLI piper natif sur RTX 3090"""
        try:
            # Commande piper native
            cmd = [
                self.piper_path,
                '--model', self.model_path,
                '--output_file', output_path,
                '--text', text
            ]
            
            print(f"🔄 Commande piper RTX 3090: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ Piper CLI RTX 3090 réussi")
                return True
            else:
                print(f"❌ Erreur piper CLI RTX 3090: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Exception piper CLI RTX 3090: {e}")
            return False
            
    def load_wav_file(self, file_path):
        """Charger fichier WAV et convertir en numpy array"""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # Convertir en numpy array
                if sample_width == 1:
                    audio_data = np.frombuffer(frames, dtype=np.uint8)
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                elif sample_width == 2:
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32767.0
                else:
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                    audio_data = audio_data.astype(np.float32) / 2147483647.0
                
                # Gérer stéréo → mono
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                return audio_data, sample_rate
                
        except Exception as e:
            print(f"❌ Erreur lecture WAV: {e}")
            return None, None
            
    def synthesize(self, text):
        """Synthèse vocale Piper NATIVE sur RTX 3090"""
        print(f"🇫🇷 Synthèse Piper NATIVE RTX 3090 (modèle original)")
        print(f"   Modèle: {self.model_path}")
        print(f"   Texte: '{text}'")
        
        try:
            # Créer fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Synthèse CLI native sur RTX 3090
            success = self.synthesize_with_cli(text, tmp_path)
            
            if success and os.path.exists(tmp_path):
                # Charger audio
                audio_data, wav_sample_rate = self.load_wav_file(tmp_path)
                
                if audio_data is not None:
                    print(f"   🎵 Audio natif RTX 3090 généré: {len(audio_data)} échantillons")
                    print(f"   📊 Sample rate: {wav_sample_rate}Hz")
                    print(f"   🔍 Range audio: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
                    
                    # Vérifier qualité
                    amplitude = max(abs(audio_data.min()), abs(audio_data.max()))
                    if amplitude > 0.01:
                        print(f"   ✅ Audio natif français RTX 3090 valide (amplitude: {amplitude:.3f})")
                    else:
                        print(f"   ⚠️ Audio natif français RTX 3090 faible (amplitude: {amplitude:.3f})")
                    
                    # Conversion pour lecture
                    audio_int16 = (audio_data * 32767).clip(-32767, 32767).astype(np.int16)
                    
                    # Nettoyage
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    return audio_int16
                    
            # Nettoyage en cas d'erreur
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            return np.array([], dtype=np.int16)
            
        except Exception as e:
            print(f"❌ Erreur synthèse native RTX 3090: {e}")
            import traceback
            traceback.print_exc()
            return np.array([], dtype=np.int16)
    
    def speak(self, text):
        """Synthèse et lecture audio Piper natif sur RTX 3090"""
        audio_data = self.synthesize(text)
        
        if len(audio_data) > 0:
            print(f"   🔊 Lecture audio natif RTX 3090...")
            sd.play(audio_data, samplerate=self.sample_rate)
            sd.wait()
            print(f"   ✅ Lecture native RTX 3090 terminée")
        else:
            print(f"   ❌ Pas d'audio natif RTX 3090 à lire")
            
        return audio_data

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory() 