# TTS/tts_handler.py
import asyncio
import tempfile
import os
import sounddevice as sd
import soundfile as sf
import edge_tts

class TTSHandler:
    def __init__(self, config):
        self.config = config
        self.voice = "fr-FR-DeniseNeural"  # Voix française premium Microsoft
        self.rate = "+0%"  # Vitesse normale
        self.volume = "+0%"  # Volume normal
        print("TTS Handler initialisé avec edge-tts (Microsoft Neural Voice).")

    def speak(self, text):
        """Synthétise et joue le texte avec edge-tts."""
        print("🔊 Synthèse vocale en cours...")
        
        # Créer un fichier temporaire pour l'audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Synthèse asynchrone avec edge-tts
            asyncio.run(self._synthesize_async(text, temp_path))
            
            # Lire et jouer l'audio
            audio_data, sample_rate = sf.read(temp_path)
            sd.play(audio_data, sample_rate)
            sd.wait()  # Attendre la fin de la lecture
            
        except Exception as e:
            print(f"Erreur TTS: {e}")
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        print("🔊 Fin de la synthèse.")

    async def _synthesize_async(self, text, output_path):
        """Méthode asynchrone pour la synthèse avec edge-tts."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path) 