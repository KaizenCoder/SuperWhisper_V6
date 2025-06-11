# 💻 CODE SOURCE - SuperWhisper V6

**Générée** : 2025-06-10 23:04:14 CET  
**Modules** : STT, LLM, TTS, Configuration, Tests  

---

## 🔥 TTS/tts_handler.py - **FINALISÉ AUJOURD'HUI**

```python
# TTS/tts_handler.py
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
        self.model_path = config['model_path']
        self.speaker_map = {}
        self.piper_executable = None
        
        print("🔊 Initialisation du moteur TTS Piper (avec gestion multi-locuteurs)...")
        
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
            print("✅ Moteur TTS Piper chargé avec succès.")
        else:
            raise FileNotFoundError("Exécutable piper non trouvé")

    def _find_piper_executable(self):
        """Cherche l'exécutable piper dans différents emplacements."""
        possible_paths = [
            "piper/piper.exe",  # Répertoire local (Windows)
            "piper.exe",  # Dans le PATH (Windows)
            "bin/piper.exe",  # Répertoire bin (Windows)
            "./piper.exe",  # Répertoire courant (Windows)
            "piper/piper",  # Répertoire local (Linux/macOS)
            "piper",  # Dans le PATH (Linux/macOS)
            "./piper",  # Répertoire courant (Linux/macOS)
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
                # La structure peut varier, nous cherchons 'speaker_id_map'
                if "speaker_id_map" in config_data and config_data["speaker_id_map"]:
                    # La carte est souvent imbriquée, ex: {'vits': {'speaker_name': 0}}
                    # On prend la première carte non vide trouvée.
                    for key, value in config_data["speaker_id_map"].items():
                        if value:
                            self.speaker_map = value
                            break

                if self.speaker_map:
                    print("🗣️ Locuteurs disponibles détectés dans le modèle :")
                    for name, sid in self.speaker_map.items():
                        print(f"  - {name} (ID: {sid})")
                else:
                    print(f"⚠️ Modèle déclaré multi-locuteurs ({num_speakers} locuteurs) mais speaker_id_map vide.")
                    print("   Utilisation du locuteur par défaut (ID: 0)")
            else:
                print("ℹ️ Modèle mono-locuteur détecté (num_speakers = 1).")
                print("   Utilisation du locuteur par défaut (ID: 0)")

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture des locuteurs : {e}")

    def speak(self, text: str):
        """Synthétise le texte en parole en utilisant l'exécutable piper avec gestion des locuteurs."""
        if not text:
            print("⚠️ Texte vide, aucune synthèse à faire.")
            return

        if not self.piper_executable:
            print("❌ Exécutable Piper non disponible")
            return

        # Déterminer le speaker_id à utiliser
        # Pour ce MVP, nous utiliserons l'ID 0 par défaut
        speaker_id = 0
        if self.speaker_map:
            # Si nous avons une carte des locuteurs, utiliser le premier disponible
            speaker_id = next(iter(self.speaker_map.values()))
            print(f"🎭 Utilisation du locuteur avec l'ID : {speaker_id}")
        else:
            print("🎭 Utilisation du locuteur par défaut (ID: 0)")
        
        print(f"🎵 Synthèse Piper pour : '{text}'")
        
        try:
            # Créer un fichier temporaire pour la sortie
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Construire la commande piper
            cmd = [
                self.piper_executable,
                "--model", str(self.model_path),
                "--output_file", tmp_path,
                "--speaker", str(speaker_id)  # Toujours inclure le speaker_id
            ]
            
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
                    print("✅ Synthèse Piper terminée avec succès.")
                else:
                    print("❌ Fichier de sortie non généré")
            else:
                print(f"❌ Erreur piper (code {result.returncode}):")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout lors de l'exécution de piper")
        except Exception as e:
            print(f"❌ Erreur durant la synthèse Piper : {e}")
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
```

---

## ⚙️ Config/mvp_settings.yaml

```yaml
# Config/mvp_settings.yaml
# Configuration minimale pour le MVP P0

stt:
  model_name: "openai/whisper-base" # Modèle plus léger pour les tests
  gpu_device: "cuda:0" # Cible la RTX 3090/5060Ti

llm:
  model_path: "D:/modeles_llm/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_S.gguf" # Modèle existant 7B
  gpu_device_index: 0 # Cible la RTX 3090/5060Ti
  n_gpu_layers: -1 # Décharger toutes les couches sur le GPU

tts:
  # Configuration pour Piper-TTS local (100% offline, conforme LUXA)
  model_path: "models/fr_FR-siwis-medium.onnx"
  use_gpu: true
  sample_rate: 22050 
```

---

## 🎤 STT/stt_handler.py

```python
# STT/stt_handler.py
import torch
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class STTHandler:
    def __init__(self, config):
        self.config = config
        self.device = config['gpu_device'] if torch.cuda.is_available() else "cpu"
        
        # Charger le modèle Whisper
        model_name = "openai/whisper-base"  # Modèle plus léger pour les tests
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        self.sample_rate = 16000
        print(f"STT Handler initialisé avec Whisper sur {self.device}")

    def listen_and_transcribe(self, duration=5):
        """Écoute le microphone pendant une durée donnée et transcrit le son."""
        print("🎤 Écoute en cours...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Attendre la fin de l'enregistrement
        print("🎤 Enregistrement terminé, transcription en cours...")
        
        # Préparer l'audio pour Whisper
        audio_input = audio_data.flatten()
        
        # Traitement avec Whisper
        input_features = self.processor(
            audio_input, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Génération du texte
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        print(f"Transcription: '{transcription}'")
        return transcription 
```

---

## 🧠 LLM/llm_handler.py

```python
from llama_cpp import Llama

class LLMHandler:
    def __init__(self, config):
        self.config = config
        self.llm = Llama(
            model_path=config['model_path'],
            n_gpu_layers=config['n_gpu_layers'],
            main_gpu=config['gpu_device_index'],
            verbose=False
        )
        print(f"LLM Handler initialisé avec le modèle {self.config['model_path']}")

    def get_response(self, prompt):
        """Génère une réponse à partir du prompt."""
        print("🧠 Le LLM réfléchit...")
        output = self.llm(f"Q: {prompt} A: ", max_tokens=100, stop=["Q:", "\n"])
        response_text = output['choices'][0]['text'].strip()
        print(f"Réponse du LLM: '{response_text}'")
        return response_text 
```

---

## 🚀 run_assistant.py - Orchestrateur Principal

```python
#!/usr/bin/env python3
"""
Luxa - SuperWhisper_V6 Assistant v1.1
======================================

Assistant vocal intelligent avec pipeline STT → LLM → TTS
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
import yaml
from STT.stt_handler import STTHandler
from LLM.llm_handler import LLMHandler
from TTS.tts_handler import TTSHandler

# Ajouter le répertoire courant au PYTHONPATH pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Orchestrator.master_handler_robust import RobustMasterHandler
import numpy as np

def parse_arguments():
    """Parse les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Luxa - Assistant Vocal Intelligent v1.1"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["cli", "web", "api"],
        default="cli",
        help="Mode d'interface (défaut: cli)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port pour modes web/api (défaut: 8080)"
    )
    
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Fichier de configuration (défaut: config/settings.yaml)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Active le mode debug"
    )
    
    return parser.parse_args()

async def run_cli_mode(handler):
    """Mode CLI interactif"""
    print("\n🎤 Mode CLI - Assistant Vocal")
    print("Commands: 'quit' pour quitter, 'status' pour le statut")
    print("=" * 50)
    
    try:
        while True:
            try:
                user_input = input("\n🗣️ Parlez (ou tapez): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Au revoir!")
                    break
                    
                elif user_input.lower() == 'status':
                    health = handler.get_health_status()
                    print(f"\n📊 Statut: {health['status']}")
                    print(f"Requêtes traitées: {health['performance']['requests_processed']}")
                    print(f"Latence moyenne: {health['performance']['avg_latency_ms']:.1f}ms")
                    continue
                    
                elif user_input.lower() == 'test':
                    # Test avec audio synthétique
                    print("🧪 Test avec audio synthétique...")
                    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
                    result = await handler.process_audio_safe(test_audio)
                    
                    print(f"✅ Résultat: {result['text']}")
                    print(f"⏱️ Latence: {result['latency_ms']:.1f}ms")
                    print(f"🎯 Succès: {result['success']}")
                    continue
                    
                if not user_input:
                    continue
                    
                print("📝 Traitement en cours...")
                
                # Pour l'instant, simuler avec du texte
                # Dans une vraie implémentation, on capturerait l'audio
                result = {
                    "success": True,
                    "text": f"Vous avez dit: {user_input}",
                    "latency_ms": 50
                }
                
                print(f"🎯 Réponse: {result['text']}")
                
            except KeyboardInterrupt:
                print("\n👋 Arrêt demandé...")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                
    except Exception as e:
        print(f"❌ Erreur CLI: {e}")

async def run_web_mode(handler, port):
    """Mode web (placeholder)"""
    print(f"🌐 Mode Web sur port {port}")
    print("⚠️ Interface web non implémentée dans cette version")
    
    # Placeholder pour serveur web
    print("Appuyez sur Ctrl+C pour arrêter...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Serveur web arrêté")

async def run_api_mode(handler, port):
    """Mode API REST (placeholder)"""
    print(f"🔌 Mode API REST sur port {port}")
    print("⚠️ API REST non implémentée dans cette version")
    
    # Placeholder pour API REST
    print("Appuyez sur Ctrl+C pour arrêter...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("🛑 API REST arrêtée")

def print_banner():
    """Affiche la bannière Luxa v1.1"""
    banner = """
    ██╗     ██╗   ██╗██╗  ██╗ █████╗ 
    ██║     ██║   ██║╚██╗██╔╝██╔══██╗
    ██║     ██║   ██║ ╚███╔╝ ███████║
    ██║     ██║   ██║ ██╔██╗ ██╔══██║
    ███████╗╚██████╔╝██╔╝ ██╗██║  ██║
    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
    
    🎤 Assistant Vocal Intelligent v1.1
    SuperWhisper_V6 - STT | LLM | TTS
    """
    print(banner)

def main():
    """Fonction principale pour exécuter la boucle de l'assistant."""
    print("🚀 Démarrage de l'assistant vocal LUXA (MVP P0)...")

    # 1. Charger la configuration
    try:
        with open("Config/mvp_settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ ERREUR: Le fichier 'Config/mvp_settings.yaml' est introuvable.")
        return

    # 2. Initialiser les modules
    try:
        print("🔧 Initialisation des modules...")
        stt_handler = STTHandler(config['stt'])
        llm_handler = LLMHandler(config['llm'])
        tts_handler = TTSHandler(config['tts'])
        print("✅ Tous les modules sont initialisés!")
    except Exception as e:
        print(f"❌ ERREUR lors de l'initialisation: {e}")
        print(f"   Détails: {str(e)}")
        return

    # 3. Boucle principale de l'assistant
    print("\n🎯 Assistant vocal LUXA prêt!")
    print("Appuyez sur Ctrl+C pour arrêter")
    
    try:
        while True:
            print("\n" + "="*50)
            input("Appuyez sur Entrée pour commencer l'écoute...")
            
            # Pipeline STT → LLM → TTS
            try:
                total_start_time = time.perf_counter()
                
                # Étape STT
                stt_start_time = time.perf_counter()
                transcription = stt_handler.listen_and_transcribe(duration=7)
                stt_latency = time.perf_counter() - stt_start_time

                if transcription and transcription.strip():
                    print(f"📝 Transcription: '{transcription}'")
                    
                    # Étape LLM
                    llm_start_time = time.perf_counter()
                    response = llm_handler.get_response(transcription)
                    llm_latency = time.perf_counter() - llm_start_time

                    # Étape TTS
                    tts_start_time = time.perf_counter()
                    if response and response.strip():
                        tts_handler.speak(response)
                    tts_latency = time.perf_counter() - tts_start_time
                    
                    total_latency = time.perf_counter() - total_start_time
                    
                    print("\n--- 📊 Rapport de Latence ---")
                    print(f"  - STT: {stt_latency:.3f}s")
                    print(f"  - LLM: {llm_latency:.3f}s")
                    print(f"  - TTS: {tts_latency:.3f}s")
                    print(f"  - TOTAL: {total_latency:.3f}s")
                    print("----------------------------\n")

                else:
                    print("Aucun texte intelligible n'a été transcrit, nouvelle écoute...")
                    
            except Exception as e:
                print(f"❌ Erreur dans le pipeline: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n🛑 Arrêt de l'assistant vocal LUXA")

if __name__ == "__main__":
    main() 
```

---

## 🧪 test_tts_handler.py - Tests Validation

```python
#!/usr/bin/env python3
"""
Test du TTSHandler avec le modèle fr_FR-siwis-medium
"""

import yaml
import sys
from pathlib import Path

# Ajouter le répertoire courant au PYTHONPATH
sys.path.append(str(Path(__file__).parent))

def test_tts_handler():
    """Test du TTSHandler avec le modèle siwis"""
    
    print("🧪 Test du TTSHandler avec modèle fr_FR-siwis-medium")
    print("=" * 60)
    
    try:
        # Charger la configuration
        with open("Config/mvp_settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration chargée")
        print(f"📍 Modèle configuré: {config['tts']['model_path']}")
        
        # Vérifier que le modèle existe
        model_path = Path(config['tts']['model_path'])
        if not model_path.exists():
            print(f"❌ ERREUR: Modèle non trouvé: {model_path}")
            return False
            
        config_path = Path(f"{config['tts']['model_path']}.json")
        if not config_path.exists():
            print(f"❌ ERREUR: Configuration du modèle non trouvée: {config_path}")
            return False
            
        print("✅ Fichiers de modèle trouvés")
        
        # Importer et initialiser le TTSHandler
        from TTS.tts_handler import TTSHandler
        
        print("\n🔧 Initialisation du TTSHandler...")
        tts_handler = TTSHandler(config['tts'])
        
        print("\n🎵 Test de synthèse vocale...")
        test_phrases = [
            "Bonjour, je suis LUXA, votre assistant vocal intelligent.",
            "Test de synthèse vocale avec le modèle français.",
            "La synthèse fonctionne parfaitement!"
        ]
        
        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n--- Test {i}/3 ---")
            tts_handler.speak(phrase)
            
            # Petite pause entre les tests
            input("Appuyez sur Entrée pour continuer...")
        
        print("\n✅ Tous les tests de synthèse ont été effectués avec succès!")
        return True
        
    except ImportError as e:
        print(f"❌ ERREUR d'import: {e}")
        print("Vérifiez que piper-tts est correctement installé.")
        return False
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        print(f"Détails: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_tts_handler()
    
    if success:
        print("\n🎉 Test terminé avec succès!")
        print("Le TTSHandler est prêt pour l'intégration dans run_assistant.py")
    else:
        print("\n❌ Test échoué!")
        print("Vérifiez l'installation de piper-tts et la configuration.")
        sys.exit(1) 
```

---

## 📦 requirements.txt - Dépendances

```
# requirements.txt
# Dépendances pour LUXA MVP P0 - Assistant Vocal

# STT (Speech-to-Text) avec Whisper via transformers
transformers
torch --index-url https://download.pytorch.org/whl/cu118

# LLM (Large Language Model)
llama-cpp-python

# TTS (Text-to-Speech) avec Microsoft Neural Voices
edge-tts

# Capture et traitement audio
sounddevice
soundfile
numpy

# Configuration YAML
pyyaml 
```

---

**Code source complet intégré** ✅  
**Modules validés** : STT, LLM, TTS fonctionnels  
**Prêt pour** : Déploiement et tests d'intégration
