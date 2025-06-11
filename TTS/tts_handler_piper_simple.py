# TTS/tts_handler_piper_simple.py
"""
TTSHandler utilisant piper-tts directement
Solution alternative pour éviter les problèmes avec piper-phonemize
"""

import json
import sys
from pathlib import Path
import tempfile
import wave
import numpy as np
import sounddevice as sd

# Tentative d'import de piper-tts
try:
    import piper
    PIPER_AVAILABLE = True
    print("✅ Module piper-tts trouvé")
except ImportError:
    PIPER_AVAILABLE = False
    print("❌ Module piper-tts non trouvé")

class TTSHandler:
    def __init__(self, config):
        self.model_path = config.get('model_path', '')
        self.speaker_map = {}
        self.piper_voice = None
        
        print("🔊 Initialisation du moteur TTS Piper (version simplifiée)...")
        
        if not PIPER_AVAILABLE:
            raise ImportError("Module piper-tts non disponible")
        
        # Vérifier que le modèle existe
        model_p = Path(self.model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Fichier modèle .onnx non trouvé : {self.model_path}")
        
        config_p = Path(f"{self.model_path}.json")
        if not config_p.exists():
            raise FileNotFoundError(f"Fichier de configuration .json non trouvé : {config_p}")

        # Charger la carte des locuteurs depuis le fichier JSON
        self._load_speaker_map(config_p)
        
        # Essayer d'initialiser Piper
        try:
            print(f"📂 Chargement du modèle : {self.model_path}")
            
            # Différentes méthodes pour charger le modèle selon la version de piper-tts
            if hasattr(piper, 'PiperVoice'):
                self.piper_voice = piper.PiperVoice.load(str(model_p))
            elif hasattr(piper, 'Voice'):
                self.piper_voice = piper.Voice(str(model_p))
            else:
                # Méthode alternative
                self.piper_voice = piper.load_model(str(model_p))
                
            print("✅ Moteur TTS Piper chargé avec succès.")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement de Piper : {e}")
            print(f"   Type d'erreur : {type(e).__name__}")
            
            # Essayer une méthode alternative
            try:
                print("🔄 Tentative de chargement alternatif...")
                # Import alternatif
                from piper import PiperVoice
                self.piper_voice = PiperVoice.load(str(model_p))
                print("✅ Chargement alternatif réussi.")
            except Exception as e2:
                print(f"❌ Échec du chargement alternatif : {e2}")
                raise e2

    def _load_speaker_map(self, config_path: Path):
        """Charge la carte des locuteurs depuis le fichier de configuration."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # Vérifier le nombre de locuteurs
            num_speakers = config_data.get("num_speakers", 1)
            
            if num_speakers > 1:
                speaker_id_map = config_data.get("speaker_id_map", {})
                if speaker_id_map:
                    if isinstance(speaker_id_map, dict):
                        for key, value in speaker_id_map.items():
                            if isinstance(value, dict) and value:
                                self.speaker_map = value
                                break
                        if not self.speaker_map and speaker_id_map:
                            self.speaker_map = speaker_id_map
                
                if self.speaker_map:
                    print("🗣️ Locuteurs disponibles détectés dans le modèle :")
                    for name, sid in self.speaker_map.items():
                        print(f"  - {name} (ID: {sid})")
                else:
                    print(f"⚠️ Modèle déclaré multi-locuteurs ({num_speakers} locuteurs) mais speaker_id_map vide.")
            else:
                print("ℹ️ Modèle mono-locuteur détecté (num_speakers = 1).")

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture des locuteurs : {e}")

    def speak(self, text: str):
        """Synthétise le texte en parole en utilisant Piper."""
        if not text:
            print("⚠️ Texte vide, aucune synthèse à faire.")
            return

        if not self.piper_voice:
            print("❌ Moteur Piper non disponible")
            return

        # Déterminer le speaker_id
        speaker_id = 0
        if self.speaker_map:
            speaker_id = next(iter(self.speaker_map.values()))
            print(f"🎭 Utilisation du locuteur avec l'ID : {speaker_id}")
        else:
            print("🎭 Utilisation du locuteur par défaut (ID: 0)")
        
        print(f"🎵 Synthèse Piper pour : '{text}'")
        
        try:
            # Créer un fichier temporaire pour la sortie
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Essayer différentes méthodes de synthèse selon l'API disponible
            audio_data = None
            
            if hasattr(self.piper_voice, 'say'):
                # Méthode directe avec lecture automatique
                try:
                    self.piper_voice.say(text, speaker_id=speaker_id)
                    print("✅ Synthèse Piper terminée (say).")
                    return
                except Exception as e:
                    print(f"⚠️ Échec de la méthode 'say': {e}")
            
            if hasattr(self.piper_voice, 'synthesize'):
                # Méthode avec fichier de sortie
                try:
                    self.piper_voice.synthesize(text, tmp_path, speaker_id=speaker_id)
                    # Lire le fichier généré et le jouer
                    self._play_wav_file(tmp_path)
                    print("✅ Synthèse Piper terminée (synthesize).")
                    return
                except Exception as e:
                    print(f"⚠️ Échec de la méthode 'synthesize': {e}")
            
            # Méthode de fallback avec génération d'audio
            if hasattr(self.piper_voice, 'generate_audio'):
                try:
                    audio_data = self.piper_voice.generate_audio(text, speaker_id=speaker_id)
                    if audio_data is not None:
                        # Jouer l'audio directement
                        sd.play(audio_data, samplerate=22050)
                        sd.wait()
                        print("✅ Synthèse Piper terminée (generate_audio).")
                        return
                except Exception as e:
                    print(f"⚠️ Échec de la méthode 'generate_audio': {e}")
            
            print("❌ Aucune méthode de synthèse fonctionnelle trouvée")
            
        except Exception as e:
            print(f"❌ Erreur durant la synthèse Piper : {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Nettoyer le fichier temporaire si créé
            try:
                if 'tmp_path' in locals():
                    Path(tmp_path).unlink(missing_ok=True)
            except:
                pass

    def _play_wav_file(self, file_path):
        """Joue un fichier WAV."""
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
                
                # Jouer l'audio
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()
                
        except Exception as e:
            print(f"❌ Erreur lecture WAV: {e}") 