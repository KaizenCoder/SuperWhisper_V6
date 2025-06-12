# TTS/tts_handler.py
import json
import sys
from pathlib import Path
import sounddevice as sd

# Ajouter le chemin vers le module piper
piper_path = Path(__file__).parent.parent / "piper" / "src" / "python_run"
sys.path.insert(0, str(piper_path))

from piper.voice import PiperVoice

class TTSHandler:
    def __init__(self, config):
        self.model_path = config['model_path']
        self.voice = None
        self.speaker_map = {}
        
        print("🔊 Initialisation du moteur TTS Piper (avec gestion multi-locuteurs)...")
        
        model_p = Path(self.model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Fichier modèle .onnx non trouvé : {self.model_path}")
        
        config_p = Path(f"{self.model_path}.json")
        if not config_p.exists():
            raise FileNotFoundError(f"Fichier de configuration .json non trouvé : {config_p}")

        # Charger la carte des locuteurs depuis le fichier JSON
        self._load_speaker_map(config_p)
        
        # Charger la voix
        self.voice = PiperVoice.load(self.model_path, config_path=str(config_p))
        print("✅ Moteur TTS Piper chargé avec succès.")

    def _load_speaker_map(self, config_path: Path):
        """Charge la carte des locuteurs depuis le fichier de configuration."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # Vérifier le nombre de locuteurs
            num_speakers = config_data.get("num_speakers", 1)
            
            if num_speakers > 1:
                # La structure peut varier, nous cherchons 'speaker_id_map'
                if "speaker_id_map" in config_data and config_data["speaker_id_map"]:
                    # La carte est souvent imbriquée, ex: {'vits': {'speaker_name': 0}}
                    # On prend la première carte non vide trouvée.
                    speaker_id_map = config_data["speaker_id_map"]
                    if isinstance(speaker_id_map, dict):
                        if speaker_id_map:
                            # Si c'est un dictionnaire imbriqué
                            for key, value in speaker_id_map.items():
                                if isinstance(value, dict) and value:
                                    self.speaker_map = value
                                    break
                            # Si c'est directement la carte
                            if not self.speaker_map and speaker_id_map:
                                self.speaker_map = speaker_id_map
                
                if self.speaker_map:
                    print("🗣️ Locuteurs disponibles détectés dans le modèle :")
                    for name, sid in self.speaker_map.items():
                        print(f"  - {name} (ID: {sid})")
                else:
                    print(f"⚠️ Modèle déclaré multi-locuteurs ({num_speakers} locuteurs) mais speaker_id_map vide.")
                    print("   Utilisation de l'ID par défaut 0.")
            else:
                print("ℹ️ Modèle mono-locuteur détecté (num_speakers = 1).")

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture des locuteurs : {e}")

    def speak(self, text: str):
        """Synthétise le texte en parole en utilisant un speaker_id."""
        if not text:
            print("⚠️ Texte vide, aucune synthèse à faire.")
            return

        # Pour le MVP, on choisit le premier locuteur disponible (généralement ID 0)
        speaker_id = 0
        if self.speaker_map:
            # Prend le premier ID de la liste
            speaker_id = next(iter(self.speaker_map.values()))
            print(f"🎭 Utilisation du locuteur avec l'ID : {speaker_id}")
        else:
            print("🎭 Utilisation du locuteur par défaut (ID: 0)")
        
        print(f"🎵 Synthèse Piper pour : '{text}'")
        try:
            # Utiliser la méthode 'say' qui gère le streaming vers les haut-parleurs
            self.voice.say(text, speaker_id=speaker_id)
            print("✅ Synthèse Piper terminée.")
        except Exception as e:
            print(f"❌ Erreur durant la synthèse Piper : {e}")
            print("   Vérifiez que le speaker_id est correct pour ce modèle.")
            # Tentative de fallback sans speaker_id pour les modèles mono-locuteurs
            try:
                print("🔄 Tentative de synthèse sans speaker_id...")
                self.voice.say(text)
                print("✅ Synthèse Piper terminée (sans speaker_id).")
            except Exception as e2:
                print(f"❌ Erreur même sans speaker_id : {e2}") 