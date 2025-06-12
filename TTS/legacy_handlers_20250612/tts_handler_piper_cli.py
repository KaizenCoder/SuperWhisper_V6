# TTS/tts_handler_piper_cli.py
"""
TTSHandler utilisant l'exécutable piper en ligne de commande
Solution de contournement pour éviter les problèmes avec piper-phonemize
"""

import json
import subprocess
import tempfile
import wave
from pathlib import Path
import numpy as np
import sounddevice as sd

class TTSHandler:
    def __init__(self, config):
        self.model_path = config.get('model_path', '')
        self.speaker_map = {}
        self.piper_executable = None
        
        print("🔊 Initialisation du moteur TTS Piper (CLI)...")
        
        # Vérifier que le modèle existe
        model_p = Path(self.model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Fichier modèle .onnx non trouvé : {self.model_path}")
        
        config_p = Path(f"{self.model_path}.json")
        if not config_p.exists():
            raise FileNotFoundError(f"Fichier de configuration .json non trouvé : {config_p}")

        # Charger la carte des locuteurs depuis le fichier JSON
        self._load_speaker_map(config_p)
        
        # Chercher l'exécutable piper
        self._find_piper_executable()
        
        if self.piper_executable:
            print(f"✅ Exécutable Piper trouvé : {self.piper_executable}")
        else:
            print("❌ Exécutable Piper non trouvé")
            print("💡 Suggestion : Téléchargez l'exécutable piper depuis https://github.com/rhasspy/piper/releases")
            raise FileNotFoundError("Exécutable piper non trouvé")

    def _find_piper_executable(self):
        """Cherche l'exécutable piper dans différents emplacements."""
        possible_paths = [
            "piper.exe",  # Dans le PATH
            "piper/piper.exe",  # Répertoire local
            "bin/piper.exe",  # Répertoire bin
            "./piper.exe",  # Répertoire courant
            "piper",  # Linux/macOS dans le PATH
            "piper/piper",  # Linux/macOS répertoire local
            "./piper",  # Linux/macOS répertoire courant
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--help"], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0:
                    self.piper_executable = path
                    return
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue

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
        """Synthétise le texte en parole en utilisant l'exécutable piper."""
        if not text:
            print("⚠️ Texte vide, aucune synthèse à faire.")
            return

        if not self.piper_executable:
            print("❌ Exécutable Piper non disponible")
            return

        # Déterminer le speaker_id
        speaker_id = 0
        if self.speaker_map:
            speaker_id = next(iter(self.speaker_map.values()))
            print(f"🎭 Utilisation du locuteur avec l'ID : {speaker_id}")
        else:
            print("🎭 Utilisation du locuteur par défaut (ID: 0)")
        
        print(f"🎵 Synthèse Piper CLI pour : '{text}'")
        
        try:
            # Créer un fichier temporaire pour la sortie
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Construire la commande piper
            cmd = [
                self.piper_executable,
                "--model", str(self.model_path),
                "--output_file", tmp_path
            ]
            
            # Ajouter le speaker si nécessaire
            if self.speaker_map:
                cmd.extend(["--speaker", str(speaker_id)])
            
            print(f"🔧 Commande : {' '.join(cmd)}")
            
            # Exécuter piper avec le texte en entrée
            result = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Lire et jouer le fichier généré
                if Path(tmp_path).exists():
                    self._play_wav_file(tmp_path)
                    print("✅ Synthèse Piper CLI terminée.")
                else:
                    print("❌ Fichier de sortie non généré")
            else:
                print(f"❌ Erreur piper (code {result.returncode}):")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout lors de l'exécution de piper")
        except Exception as e:
            print(f"❌ Erreur durant la synthèse Piper CLI : {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Nettoyer le fichier temporaire
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