#!/usr/bin/env python3
"""
Gestionnaire de Chemins de Modèles - SuperWhisper V6
===================================================
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:0 après mapping) OBLIGATOIRE

Centralise la gestion des chemins vers tous les modèles IA.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
import yaml

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB sur Bus PCI 1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Force l'ordre du bus physique
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:0 après mapping)")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch

def validate_rtx3090_mandatory():
    """Validation obligatoire RTX 3090"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA non disponible - mode validation uniquement")
        return False
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    return True

class ModelPathManager:
    """Gestionnaire centralisé des chemins de modèles"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Validation RTX 3090 obligatoire
        validate_rtx3090_mandatory()
        
        # Chemin du fichier de configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "model_paths.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        print(f"📁 Configuration chargée: {self.config_path}")
    
    def _load_config(self) -> Dict:
        """Charge la configuration des chemins"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"⚠️ Configuration non trouvée: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            print(f"❌ Erreur chargement config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            "llm_models": {
                "base_directory": "D:/modeles_llm",
                "chat_models": {
                    "hermes_7b": "D:/modeles_llm/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_S.gguf"
                }
            },
            "tts_models": {
                "base_directory": "D:/TTS_Voices",
                "piper_voices": {
                    "base_path": "D:/TTS_Voices/piper",
                    "french_voices": {
                        "siwis": "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx"
                    }
                }
            },
            "ai_models": {
                "base_directory": "D:/modeles_ia"
            }
        }
    
    def get_llm_model_path(self, model_name: str) -> Optional[Path]:
        """Récupère le chemin d'un modèle LLM"""
        chat_models = self.config.get("llm_models", {}).get("chat_models", {})
        coding_models = self.config.get("llm_models", {}).get("coding_models", {})
        
        # Chercher dans les modèles de chat
        if model_name in chat_models:
            path = Path(chat_models[model_name])
            if path.exists():
                return path
            print(f"⚠️ Modèle LLM non trouvé: {path}")
        
        # Chercher dans les modèles de code
        if model_name in coding_models:
            path = Path(coding_models[model_name])
            if path.exists():
                return path
            print(f"⚠️ Modèle LLM non trouvé: {path}")
        
        print(f"❌ Modèle LLM inconnu: {model_name}")
        return None
    
    def get_tts_voice_path(self, voice_name: str) -> Optional[Path]:
        """Récupère le chemin d'une voix TTS"""
        piper_voices = self.config.get("tts_models", {}).get("piper_voices", {}).get("french_voices", {})
        
        if voice_name in piper_voices:
            path = Path(piper_voices[voice_name])
            if path.exists():
                return path
            print(f"⚠️ Voix TTS non trouvée: {path}")
        
        print(f"❌ Voix TTS inconnue: {voice_name}")
        return None
    
    def list_available_llm_models(self) -> List[str]:
        """Liste tous les modèles LLM disponibles"""
        models = []
        
        chat_models = self.config.get("llm_models", {}).get("chat_models", {})
        coding_models = self.config.get("llm_models", {}).get("coding_models", {})
        
        for name, path in chat_models.items():
            if Path(path).exists():
                models.append(f"chat/{name}")
        
        for name, path in coding_models.items():
            if Path(path).exists():
                models.append(f"coding/{name}")
        
        return models
    
    def list_available_tts_voices(self) -> List[str]:
        """Liste toutes les voix TTS disponibles"""
        voices = []
        
        piper_voices = self.config.get("tts_models", {}).get("piper_voices", {}).get("french_voices", {})
        
        for name, path in piper_voices.items():
            if Path(path).exists():
                voices.append(f"piper/{name}")
        
        return voices
    
    def validate_all_paths(self) -> Dict[str, bool]:
        """Valide que tous les chemins existent"""
        results = {}
        
        # Valider modèles LLM
        chat_models = self.config.get("llm_models", {}).get("chat_models", {})
        for name, path in chat_models.items():
            results[f"llm/{name}"] = Path(path).exists()
        
        # Valider voix TTS
        piper_voices = self.config.get("tts_models", {}).get("piper_voices", {}).get("french_voices", {})
        for name, path in piper_voices.items():
            results[f"tts/{name}"] = Path(path).exists()
        
        return results
    
    def get_gpu_config(self) -> Dict:
        """Récupère la configuration GPU"""
        return self.config.get("gpu_config", {
            "primary_device": "cuda:0",  # RTX 3090 après mapping
            "memory_optimization": True,
            "precision": "float16"
        })

def main():
    """Test du gestionnaire de chemins"""
    print("🚀 TEST GESTIONNAIRE CHEMINS MODÈLES")
    print("=" * 50)
    
    # Initialiser le gestionnaire
    manager = ModelPathManager()
    
    # Lister les modèles disponibles
    print("📋 MODÈLES LLM DISPONIBLES:")
    llm_models = manager.list_available_llm_models()
    for model in llm_models:
        print(f"   ✅ {model}")
    
    print("\n🎤 VOIX TTS DISPONIBLES:")
    tts_voices = manager.list_available_tts_voices()
    for voice in tts_voices:
        print(f"   ✅ {voice}")
    
    # Valider tous les chemins
    print("\n🔍 VALIDATION DES CHEMINS:")
    validation_results = manager.validate_all_paths()
    for path, exists in validation_results.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {path}")
    
    # Tester l'accès aux modèles
    print("\n🧪 TEST ACCÈS MODÈLES:")
    
    # Test modèle LLM par défaut
    hermes_path = manager.get_llm_model_path("hermes_7b")
    if hermes_path:
        print(f"   ✅ Hermes 7B: {hermes_path}")
    
    # Test voix TTS par défaut
    siwis_path = manager.get_tts_voice_path("siwis")
    if siwis_path:
        print(f"   ✅ Voix Siwis: {siwis_path}")
    
    # Configuration GPU
    gpu_config = manager.get_gpu_config()
    print(f"\n🎮 CONFIG GPU: {gpu_config}")

if __name__ == "__main__":
    # 🚨 CRITIQUE: Configuration dual-GPU RTX 5060 (CUDA:0) + RTX 3090 (CUDA:1)
    # RTX 5060 (CUDA:0) = INTERDITE (8GB insuffisant)  
    # RTX 3090 (CUDA:1) = SEULE AUTORISÉE (24GB VRAM)
    
    main()
