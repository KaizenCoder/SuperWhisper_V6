# TTS/tts_handler_piper.py
import os
import tempfile
import sounddevice as sd
import soundfile as sf
import piper

class TTSHandlerPiper:
    def __init__(self, config):
        self.config = config
        
        # Chemin vers le modèle français local
        self.model_path = config.get('model_path', './models/fr_FR-siwis-medium.onnx')
        self.use_gpu = config.get('use_gpu', True)
        
        # Charger le modèle Piper
        try:
            self.voice = piper.PiperVoice.load(self.model_path, use_cuda=self.use_gpu)
            print(f"✅ TTS Handler Piper initialisé - Modèle: {self.model_path}")
            print(f"🚀 GPU activé: {self.use_gpu}")
        except Exception as e:
            print(f"❌ Erreur chargement modèle Piper: {e}")
            raise

    def speak(self, text):
        """Synthétise et joue le texte avec Piper (100% local)."""
        print("🔊 Synthèse vocale Piper en cours...")
        
        try:
            # Synthèse avec Piper (local)
            audio_bytes = self.voice.synthesize(text)
            
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name
            
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
            print(f"❌ Erreur synthèse Piper: {e}")
            
        print("🔊 Fin de la synthèse.")

    def test_synthesis(self):
        """Test rapide de synthèse."""
        test_text = "Bonjour, je suis LUXA, votre assistant vocal local."
        print(f"🧪 Test de synthèse: '{test_text}'")
        self.speak(test_text) 