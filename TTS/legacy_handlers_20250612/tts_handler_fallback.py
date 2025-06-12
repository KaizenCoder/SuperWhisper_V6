"""
TTSHandler de fallback utilisant Windows SAPI
Utilisé temporairement en attendant que Piper soit correctement installé
"""

import json
from pathlib import Path
import win32com.client

class TTSHandler:
    def __init__(self, config):
        self.model_path = config.get('model_path', '')
        self.speaker_map = {}
        
        print("🔊 Initialisation du moteur TTS SAPI (fallback temporaire)...")
        print("⚠️ ATTENTION: Utilisation de SAPI en attendant Piper")
        
        # Initialiser SAPI
        try:
            self.sapi = win32com.client.Dispatch("SAPI.SpVoice")
            
            # Essayer de trouver une voix française
            voices = self.sapi.GetVoices()
            french_voice = None
            
            for i in range(voices.Count):
                voice = voices.Item(i)
                voice_name = voice.GetDescription()
                print(f"   Voix disponible: {voice_name}")
                
                # Chercher une voix française
                if any(keyword in voice_name.lower() for keyword in ['french', 'français', 'france', 'fr-']):
                    french_voice = voice
                    print(f"   ✅ Voix française trouvée: {voice_name}")
                    break
            
            if french_voice:
                self.sapi.Voice = french_voice
            else:
                print("   ⚠️ Aucune voix française trouvée, utilisation de la voix par défaut")
                
            # Simuler la lecture du modèle pour compatibilité
            if self.model_path:
                model_p = Path(self.model_path)
                config_p = Path(f"{self.model_path}.json")
                
                if config_p.exists():
                    self._load_speaker_map(config_p)
                    
            print("✅ Moteur TTS SAPI initialisé (fallback temporaire)")
            
        except Exception as e:
            print(f"❌ Erreur initialisation SAPI: {e}")
            self.sapi = None

    def _load_speaker_map(self, config_path: Path):
        """Charge la carte des locuteurs depuis le fichier de configuration (pour compatibilité)."""
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
                    print("🗣️ Locuteurs détectés dans le modèle Piper (simulation):")
                    for name, sid in self.speaker_map.items():
                        print(f"  - {name} (ID: {sid})")
                else:
                    print(f"⚠️ Modèle déclaré multi-locuteurs ({num_speakers} locuteurs) mais speaker_id_map vide.")
            else:
                print("ℹ️ Modèle mono-locuteur détecté (simulation).")

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture des locuteurs : {e}")

    def speak(self, text: str):
        """Synthétise le texte en parole en utilisant SAPI (fallback)."""
        if not text:
            print("⚠️ Texte vide, aucune synthèse à faire.")
            return

        if not self.sapi:
            print("❌ Moteur SAPI non disponible")
            return

        # Afficher les informations de compatibilité Piper
        speaker_id = 0
        if self.speaker_map:
            speaker_id = next(iter(self.speaker_map.values()))
            print(f"🎭 Simulation locuteur Piper ID : {speaker_id}")
        else:
            print("🎭 Simulation locuteur par défaut (ID: 0)")
        
        print(f"🎵 Synthèse SAPI (fallback) pour : '{text}'")
        try:
            # Utiliser SAPI pour la synthèse
            self.sapi.Speak(text)
            print("✅ Synthèse SAPI terminée.")
        except Exception as e:
            print(f"❌ Erreur durant la synthèse SAPI : {e}") 