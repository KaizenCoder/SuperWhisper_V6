#!/usr/bin/env python3
"""
TTS/tts_handler_coqui.py
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Handler TTS Coqui avec configuration GPU RTX 3090 exclusive
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 Ti (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Maintenant imports normaux...
import tempfile
import sounddevice as sd
import soundfile as sf

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
        print("⚠️ PyTorch non disponible - validation GPU ignorée pour TTS")

# Import spécifique pour éviter conflit avec dossier TTS local
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from TTS.api import TTS
except ImportError:
    print("⚠️ Coqui TTS non disponible")
    TTS = None

class TTSHandlerCoqui:
    def __init__(self, config):
        # Validation RTX 3090 obligatoire
        validate_rtx3090_mandatory()
        
        self.config = config
        
        # Modèle français local Coqui-TTS
        self.model_name = config.get('model_name', 'tts_models/fr/css10/vits')
        # RTX 3090 seule visible = utilisation automatique sécurisée
        self.device = 'cuda' if config.get('use_gpu', True) else 'cpu'  # RTX 3090 automatiquement
        
        if TTS is None:
            print("❌ Coqui TTS non disponible - Handler non fonctionnel")
            return
        
        try:
            # Initialiser Coqui-TTS (100% local) sur RTX 3090
            self.tts = TTS(model_name=self.model_name).to(self.device)
            print(f"✅ TTS Handler Coqui RTX 3090 initialisé")
            print(f"📦 Modèle: {self.model_name}")
            print(f"🚀 Device RTX 3090: {self.device}")
        except Exception as e:
            print(f"❌ Erreur chargement modèle Coqui RTX 3090: {e}")
            raise

    def speak(self, text):
        """Synthétise et joue le texte avec Coqui-TTS (100% local sur RTX 3090)."""
        print("🔊 Synthèse vocale Coqui RTX 3090 en cours...")
        
        if TTS is None or not hasattr(self, 'tts'):
            print("❌ TTS Coqui non initialisé")
            return
        
        try:
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Synthèse avec Coqui-TTS (local sur RTX 3090)
            self.tts.tts_to_file(text=text, file_path=temp_path)
            
            try:
                # Lire et jouer l'audio
                audio_data, sample_rate = sf.read(temp_path)
                sd.play(audio_data, sample_rate)
                sd.wait()  # Attendre la fin de la lecture
                
            finally:
                # Nettoyer le fichier temporaire
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f"❌ Erreur synthèse Coqui RTX 3090: {e}")
            
        print("🔊 Fin de la synthèse RTX 3090.")

    def test_synthesis(self):
        """Test rapide de synthèse sur RTX 3090."""
        test_text = "Bonjour, je suis LUXA, votre assistant vocal local et confidentiel sur RTX 3090."
        print(f"🧪 Test de synthèse RTX 3090: '{test_text}'")
        self.speak(test_text)

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory() 