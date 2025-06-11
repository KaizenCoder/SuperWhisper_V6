#!/usr/bin/env python3
"""
Fallback Manager - Luxa v1.1
=============================
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Gestionnaire de fallback intelligent avec basculement automatique selon les métriques.
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
import yaml
import torch
import time
from typing import Dict, Any, Optional
from pathlib import Path

def validate_rtx3090_mandatory():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

# Import du GPU Manager
sys.path.append(str(Path(__file__).parent.parent))
from utils.gpu_manager import get_gpu_manager

class FallbackManager:
    def __init__(self, config_path: str = "config/fallbacks.yaml"):
        # Validation RTX 3090 obligatoire
        validate_rtx3090_mandatory()
        
        self.config_path = config_path
        self.config = self._load_config()
        self.active_components = {}
        self.gpu_manager = get_gpu_manager()
        self.performance_history = {}
        
        print(f"🔄 Fallback Manager RTX 3090 initialisé")
        print(f"📋 Configuration: {config_path}")
        
    def _load_config(self) -> dict:
        """Charge la configuration de fallback"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"✅ Configuration chargée: {self.config_path}")
                return config
            else:
                print(f"⚠️ Config introuvable, utilisation config par défaut")
                return self._get_default_config()
        except Exception as e:
            print(f"❌ Erreur chargement config: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> dict:
        """Configuration par défaut si fichier absent"""
        return {
            "fallback_config": {
                "stt": {
                    "primary": "large-v3",
                    "fallback": "base",
                    "trigger": [
                        {"type": "latency", "threshold_ms": 500},
                        {"type": "vram", "threshold_gb": 2.0},
                        {"type": "exception", "exception": "OutOfMemoryError"}
                    ]
                },
                "llm": {
                    "primary": "llama-2-13b-chat.Q5_K_M.gguf",
                    "fallback": "phi-2.gguf",
                    "trigger": [
                        {"type": "latency", "threshold_ms": 2000},
                        {"type": "vram", "threshold_gb": 4.0},
                        {"type": "exception", "exception": "OutOfMemoryError"}
                    ]
                },
                "tts": {
                    "primary": "xtts-v2",
                    "fallback": "espeak",
                    "trigger": [
                        {"type": "latency", "threshold_ms": 1000},
                        {"type": "vram", "threshold_gb": 1.0},
                        {"type": "exception", "exception": "OutOfMemoryError"}
                    ]
                }
            }
        }
        
    def get_component(self, component_type: str, metrics: Optional[Dict[str, Any]] = None):
        """Retourne le composant actif ou bascule sur fallback si nécessaire"""
        
        # Enregistrer les métriques pour historique
        if metrics:
            self._record_performance(component_type, metrics)
        
        # Premier appel : charger le composant principal
        if component_type not in self.active_components:
            print(f"🚀 Chargement initial {component_type} sur RTX 3090")
            self.active_components[component_type] = {
                "component": self._load_primary(component_type),
                "type": "primary",
                "load_time": time.time()
            }
            
        # Vérifier si fallback nécessaire
        if metrics and self._should_fallback(component_type, metrics):
            current_type = self.active_components[component_type]["type"]
            
            if current_type == "primary":
                print(f"⚠️ Basculement {component_type} vers fallback RTX 3090")
                fallback_component = self._load_fallback(component_type)
                
                if fallback_component is not None:
                    # Nettoyer ancien composant si possible
                    self._cleanup_component(component_type)
                    
                    self.active_components[component_type] = {
                        "component": fallback_component,
                        "type": "fallback",
                        "load_time": time.time()
                    }
                else:
                    print(f"❌ Échec chargement fallback {component_type} RTX 3090")
            else:
                print(f"⚠️ Déjà en fallback pour {component_type} RTX 3090")
            
        return self.active_components[component_type]["component"]
        
    def _should_fallback(self, component_type: str, metrics: Dict[str, Any]) -> bool:
        """Détermine si on doit basculer sur fallback"""
        
        if component_type not in self.config["fallback_config"]:
            return False
            
        triggers = self.config["fallback_config"][component_type]["trigger"]
        
        for trigger in triggers:
            if trigger["type"] == "latency":
                if metrics.get("latency_ms", 0) > trigger["threshold_ms"]:
                    print(f"🔴 Trigger latence: {metrics.get('latency_ms'):.1f}ms > {trigger['threshold_ms']}ms")
                    return True
                    
            elif trigger["type"] == "vram":
                # RTX 3090 seule visible = device index 0
                if torch.cuda.is_available():
                    free, _ = torch.cuda.mem_get_info(0)  # RTX 3090 = index 0
                    free_gb = free / 1024**3
                    if free_gb < trigger["threshold_gb"]:
                        print(f"🔴 Trigger VRAM RTX 3090: {free_gb:.1f}GB < {trigger['threshold_gb']}GB")
                        return True
                        
            elif trigger["type"] == "exception":
                if metrics.get("exception_type") == trigger["exception"]:
                    print(f"🔴 Trigger exception: {trigger['exception']}")
                    return True
                    
        return False
        
    def _load_primary(self, component_type: str):
        """Charge le composant principal"""
        if component_type not in self.config["fallback_config"]:
            print(f"❌ Pas de config pour {component_type}")
            return None
            
        config = self.config["fallback_config"][component_type]
        
        try:
            if component_type == "stt":
                return self._load_stt_model(config["primary"])
            elif component_type == "llm":
                return self._load_llm_model(config["primary"])
            elif component_type == "tts":
                return self._load_tts_model(config["primary"])
        except Exception as e:
            print(f"❌ Erreur chargement {component_type} principal RTX 3090: {e}")
            return None
            
    def _load_fallback(self, component_type: str):
        """Charge le composant de fallback"""
        if component_type not in self.config["fallback_config"]:
            print(f"❌ Pas de config fallback pour {component_type}")
            return None
            
        config = self.config["fallback_config"][component_type]
        
        try:
            if component_type == "stt":
                return self._load_stt_model(config["fallback"], is_fallback=True)
            elif component_type == "llm":
                return self._load_llm_model(config["fallback"], is_fallback=True)
            elif component_type == "tts":
                return self._load_tts_model(config["fallback"], is_fallback=True)
        except Exception as e:
            print(f"❌ Erreur chargement {component_type} fallback RTX 3090: {e}")
            return None
            
    def _load_stt_model(self, model_name: str, is_fallback: bool = False):
        """Charge un modèle STT avec gestion d'erreur"""
        print(f"🎤 Chargement STT RTX 3090: {model_name} ({'fallback' if is_fallback else 'primary'})")
        
        try:
            if "whisper" in model_name.lower() and not is_fallback:
                try:
                    from faster_whisper import WhisperModel
                    # RTX 3090 seule visible = device index 0
                    
                    model = WhisperModel(
                        model_name,
                        device="cuda",  # RTX 3090 automatiquement
                        device_index=0,  # RTX 3090 = index 0
                        compute_type="float16"
                    )
                    print(f"✅ Whisper RTX 3090 chargé: {model_name}")
                    return model
                except ImportError:
                    print(f"⚠️ faster-whisper non disponible, utilisation alternative")
                    return None
                    
            else:
                # Fallback simple ou modèle léger
                print(f"🔄 Modèle STT simple: {model_name}")
                return {"type": "simple", "model": model_name}
                
        except Exception as e:
            print(f"❌ Erreur STT RTX 3090: {e}")
            return None
    
    def _load_llm_model(self, model_name: str, is_fallback: bool = False):
        """Charge un modèle LLM avec gestion d'erreur"""
        print(f"🧠 Chargement LLM RTX 3090: {model_name} ({'fallback' if is_fallback else 'primary'})")
        
        try:
            if not is_fallback:
                try:
                    from llama_cpp import Llama
                    
                    # Configuration RTX 3090 optimisée
                    model = Llama(
                        model_path=f"models/{model_name}",
                        n_gpu_layers=35,  # RTX 3090 24GB
                        main_gpu=0,  # RTX 3090 = index 0
                        verbose=False
                    )
                    print(f"✅ LLM RTX 3090 chargé: {model_name}")
                    return model
                except Exception as e:
                    print(f"⚠️ Erreur LLM principal: {e}")
                    return None
            else:
                # Fallback LLM léger
                print(f"🔄 LLM fallback RTX 3090: {model_name}")
                return {"type": "simple", "model": model_name}
                
        except Exception as e:
            print(f"❌ Erreur LLM RTX 3090: {e}")
            return None
    
    def _load_tts_model(self, model_name: str, is_fallback: bool = False):
        """Charge un modèle TTS avec gestion d'erreur"""
        print(f"🔊 Chargement TTS RTX 3090: {model_name} ({'fallback' if is_fallback else 'primary'})")
        
        try:
            if not is_fallback and "xtts" in model_name.lower():
                try:
                    from TTS.api import TTS
                    
                    # XTTS sur RTX 3090
                    model = TTS(model_name).to("cuda")  # RTX 3090 automatiquement
                    print(f"✅ XTTS RTX 3090 chargé: {model_name}")
                    return model
                except Exception as e:
                    print(f"⚠️ Erreur XTTS: {e}")
                    return None
            else:
                # Fallback TTS simple (espeak, etc.)
                print(f"🔄 TTS fallback: {model_name}")
                return {"type": "simple", "model": model_name}
                
        except Exception as e:
            print(f"❌ Erreur TTS RTX 3090: {e}")
            return None
    
    def _cleanup_component(self, component_type: str):
        """Nettoie un composant avant le changement"""
        if component_type in self.active_components:
            component = self.active_components[component_type]["component"]
            
            # Cleanup spécifique selon le type
            try:
                if hasattr(component, 'cleanup'):
                    component.cleanup()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Nettoyage VRAM RTX 3090
            except Exception as e:
                print(f"⚠️ Erreur cleanup {component_type}: {e}")
                
            print(f"🧹 Composant {component_type} nettoyé")
    
    def _record_performance(self, component_type: str, metrics: Dict[str, Any]):
        """Enregistre les métriques de performance"""
        if component_type not in self.performance_history:
            self.performance_history[component_type] = []
            
        self.performance_history[component_type].append({
            "timestamp": time.time(),
            "metrics": metrics.copy()
        })
        
        # Garder seulement les 100 dernières mesures
        if len(self.performance_history[component_type]) > 100:
            self.performance_history[component_type] = self.performance_history[component_type][-100:]
    
    def get_performance_stats(self, component_type: str) -> Dict[str, Any]:
        """Retourne les statistiques de performance"""
        if component_type not in self.performance_history:
            return {"error": "No data available"}
            
        history = self.performance_history[component_type]
        if not history:
            return {"error": "No metrics recorded"}
            
        # Calculer moyennes
        latencies = [m["metrics"].get("latency_ms", 0) for m in history]
        
        return {
            "count": len(history),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "last_update": history[-1]["timestamp"]
        }
    
    def force_fallback(self, component_type: str):
        """Force le basculement vers fallback"""
        print(f"🔄 Basculement forcé vers fallback: {component_type}")
        self.get_component(component_type, {"force_fallback": True})
    
    def reset_component(self, component_type: str):
        """Remet à zéro un composant (retour au principal)"""
        if component_type in self.active_components:
            self._cleanup_component(component_type)
            del self.active_components[component_type]
            print(f"🔄 Composant {component_type} remis à zéro")
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut complet du manager"""
        status = {
            "active_components": {},
            "performance_stats": {},
            "gpu_status": "rtx3090_exclusive"
        }
        
        for comp_type, comp_data in self.active_components.items():
            status["active_components"][comp_type] = {
                "type": comp_data["type"],
                "loaded": comp_data["component"] is not None,
                "load_time": comp_data["load_time"],
                "uptime_seconds": time.time() - comp_data["load_time"]
            }
            
        for comp_type in self.performance_history:
            status["performance_stats"][comp_type] = self.get_performance_stats(comp_type)
            
        # Statut GPU RTX 3090
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)  # RTX 3090 = index 0
            status["gpu_memory"] = {
                "free_gb": free / 1024**3,
                "total_gb": total / 1024**3,
                "used_percent": ((total - free) / total) * 100
            }
            
        return status

def test_fallback_manager():
    """Test du Fallback Manager"""
    print("🧪 Test Fallback Manager RTX 3090")
    
    manager = FallbackManager()
    
    # Test STT
    stt_model = manager.get_component("stt")
    print(f"STT: {stt_model}")
    
    # Test avec métriques dégradées
    bad_metrics = {"latency_ms": 1000, "exception_type": "OutOfMemoryError"}
    stt_fallback = manager.get_component("stt", bad_metrics)
    print(f"STT Fallback: {stt_fallback}")
    
    # Statut
    status = manager.get_status()
    print(f"Status: {status}")

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    test_fallback_manager() 