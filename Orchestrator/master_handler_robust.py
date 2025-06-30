#!/usr/bin/env python3
"""
Master Handler Robuste - Luxa v1.1 - VERSION AMÉLIORÉE
========================================================

Pipeline principal avec sécurité intégrée, gestion d'erreurs robuste,
et circuit breakers pour tous les composants critiques.

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import pathlib

# =============================================================================
# 🚀 PORTABILITÉ AUTOMATIQUE - EXÉCUTABLE DEPUIS N'IMPORTE OÙ
# =============================================================================
def _setup_portable_environment():
    """Configure l'environnement pour exécution portable"""
    # Déterminer le répertoire racine du projet
    current_file = pathlib.Path(__file__).resolve()
    
    # Chercher le répertoire racine (contient .git ou marqueurs projet)
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    # Ajouter le projet root au Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Changer le working directory vers project root
    os.chdir(project_root)
    
    # Configuration GPU RTX 3090 obligatoire
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
    
    print(f"🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
    print(f"📁 Project Root: {project_root}")
    print(f"💻 Working Directory: {os.getcwd()}")
    
    return project_root

# Initialiser l'environnement portable
_PROJECT_ROOT = _setup_portable_environment()

# Maintenant imports normaux...

import time
import torch
import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys

# Imports des modules Luxa
sys.path.append(str(Path(__file__).parent.parent))
from utils.gpu_manager import get_gpu_manager
from Orchestrator.fallback_manager import FallbackManager
from monitoring.prometheus_exporter_enhanced import EnhancedMetricsCollector
from STT.vad_manager import OptimizedVADManager

# NOUVELLES IMPORTATIONS SÉCURITÉ ET ROBUSTESSE
from config.security_config import SecurityConfig
from utils.error_handler import RobustErrorHandler, protect_component, CircuitOpenError

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobustMasterHandler:
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialise le gestionnaire principal robuste avec sécurité"""
        
        print("🚀 Initialisation Master Handler Robuste v1.1...")
        
        # ✅ NOUVEAUTÉ: Composants de sécurité et robustesse
        self.security_config = SecurityConfig()
        self.error_handler = RobustErrorHandler()
        
        # Enregistrer les composants critiques avec circuit breakers
        self.error_handler.register_component("stt", failure_threshold=3, max_retries=2)
        self.error_handler.register_component("vad", failure_threshold=5, max_retries=1)
        self.error_handler.register_component("llm", failure_threshold=2, max_retries=3)
        self.error_handler.register_component("tts", failure_threshold=3, max_retries=2)
        
        # Composants de base (existants)
        self.gpu_manager = get_gpu_manager()
        self.fallback_manager = FallbackManager()
        self.metrics = EnhancedMetricsCollector(port=8000)
        self.vad_manager = None
        
        # État du pipeline
        self.components = {}
        self.is_initialized = False
        self.error_counts = {"stt": 0, "llm": 0, "tts": 0, "vad": 0}
        self.last_errors = {}
        
        # Statistiques de performance
        self.performance_stats = {
            "requests_processed": 0,
            "total_latency_ms": 0,
            "avg_latency_ms": 0,
            "success_rate": 0.0,
            "errors_by_component": {}
        }
        
        print("✅ Gestionnaire principal initialisé avec sécurité renforcée")
    
    # ✅ NOUVEAUTÉ: Méthode d'authentification sécurisée
    async def authenticate_request(self, api_key: Optional[str] = None, 
                                 jwt_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Authentifie une requête via API key ou JWT token
        
        Args:
            api_key: Clé API optionnelle
            jwt_token: Token JWT optionnel
            
        Returns:
            Dict avec informations utilisateur si authentifié
            
        Raises:
            Exception: Si authentification échoue
        """
        
        if api_key:
            user = self.security_config.validate_api_key(api_key)
            if user:
                return {"username": user, "auth_method": "api_key"}
        
        if jwt_token:
            user_data = self.security_config.validate_jwt_token(jwt_token)
            if user_data:
                return {**user_data, "auth_method": "jwt"}
        
        raise Exception("Authentification requise - clé API ou token JWT invalide")
    
    # ✅ NOUVEAUTÉ: Validation sécurisée des entrées audio
    def validate_audio_input(self, audio_data: np.ndarray, 
                           filename: str = "audio.wav") -> Dict[str, Any]:
        """
        Valide les données audio avec contrôles sécurité
        
        Args:
            audio_data: Données audio numpy
            filename: Nom de fichier pour validation format
            
        Returns:
            Dict avec résultat validation
        """
        
        # Conversion en bytes pour validation
        if audio_data.dtype != np.int16:
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        else:
            audio_bytes = audio_data.tobytes()
        
        # Validation sécurité
        validation = self.security_config.validate_audio_input(audio_bytes, filename)
        
        # Validations supplémentaires spécifiques audio
        if len(audio_data) > 48000 * 30:  # Max 30 secondes à 48kHz
            validation['valid'] = False
            validation['errors'].append("Audio trop long (max 30s)")
        
        if np.max(np.abs(audio_data)) > 1.0:
            validation['warnings'] = validation.get('warnings', [])
            validation['warnings'].append("Audio potentiellement saturé")
        
        return validation

    async def initialize(self):
        """Initialise tous les composants avec protection d'erreur"""
        if self.is_initialized:
            return
            
        print("🔧 Initialisation des composants...")
        
        # Initialiser VAD avec protection
        try:
            await self._initialize_vad_protected()
        except Exception as e:
            logger.warning(f"⚠️ VAD non disponible: {e}")
        
        # Initialiser monitoring
        try:
            self._start_monitoring()
        except Exception as e:
            logger.warning(f"⚠️ Monitoring limité: {e}")
        
        # ✅ NOUVEAUTÉ: Pré-charger STT avec circuit breaker
        await self._preload_stt_protected()
        
        self.is_initialized = True
        print("✅ Initialisation terminée")

    # ✅ NOUVEAUTÉ: Initialisation VAD protégée
    async def _initialize_vad_protected(self):
        """Initialise VAD avec protection circuit breaker"""
        
        async def init_vad():
            self.vad_manager = OptimizedVADManager(chunk_ms=160, latency_threshold_ms=25)
            await self.vad_manager.initialize()
            return self.vad_manager
        
        try:
            self.vad_manager = await self.error_handler.execute_safe("vad", init_vad)
            print("✅ VAD initialisé avec protection")
        except CircuitOpenError:
            logger.warning("⚠️ Circuit VAD ouvert, mode fallback")
            self.vad_manager = None
        except Exception as e:
            logger.error(f"❌ Erreur initialisation VAD: {e}")
            self.vad_manager = None

    # ✅ NOUVEAUTÉ: Pré-chargement STT protégé
    async def _preload_stt_protected(self):
        """Pré-charge STT avec circuit breaker"""
        
        async def preload_stt():
            start_time = time.time()
            stt_component = self.fallback_manager.get_component("stt")
            load_time = time.time() - start_time
            
            if stt_component:
                self.metrics.record_model_load_time("stt", "primary", load_time)
                self.metrics.set_component_status("stt", "primary", True)
                return stt_component
            else:
                raise Exception("Aucun modèle STT disponible")
        
        try:
            await self.error_handler.execute_safe("stt", preload_stt)
            print("✅ STT pré-chargé avec protection")
        except CircuitOpenError:
            logger.warning("⚠️ Circuit STT ouvert")
        except Exception as e:
            logger.warning(f"⚠️ Pré-chargement STT échoué: {e}")

    # ✅ MÉTHODE PRINCIPALE AMÉLIORÉE avec sécurité
    async def process_audio_secure(self, 
                                  audio_chunk: np.ndarray,
                                  api_key: Optional[str] = None,
                                  jwt_token: Optional[str] = None,
                                  filename: str = "audio.wav") -> Dict[str, Any]:
        """
        Traite l'audio avec authentification et validation sécurisée
        
        Args:
            audio_chunk: Array numpy contenant l'audio (16kHz, mono)
            api_key: Clé API pour authentification
            jwt_token: Token JWT pour authentification
            filename: Nom de fichier pour validation
            
        Returns:
            Dict contenant le résultat du traitement et les métriques
        """
        
        pipeline_start = time.perf_counter()
        
        # Initialiser résultat
        result = {
            "success": False,
            "text": "",
            "confidence": 0.0,
            "latency_ms": 0,
            "components_used": {},
            "errors": [],
            "warnings": [],
            "metrics": {},
            "security": {}
        }
        
        try:
            # ✅ ÉTAPE 1: Authentification
            try:
                auth_info = await self.authenticate_request(api_key, jwt_token)
                result["security"]["authenticated"] = True
                result["security"]["user"] = auth_info.get("username", "unknown")
                result["security"]["auth_method"] = auth_info.get("auth_method")
                logger.info(f"🔐 Requête authentifiée: {auth_info['username']}")
            except Exception as e:
                result["errors"].append(f"Authentification échouée: {e}")
                result["security"]["authenticated"] = False
                return result
            
            # ✅ ÉTAPE 2: Validation audio sécurisée
            validation = self.validate_audio_input(audio_chunk, filename)
            result["security"]["audio_validation"] = validation
            
            if not validation['valid']:
                result["errors"].extend(validation['errors'])
                return result
            
            if validation.get('warnings'):
                result["warnings"].extend(validation['warnings'])
            
            # ✅ ÉTAPE 3: Initialisation si nécessaire
            if not self.is_initialized:
                await self.initialize()
            
            # ✅ ÉTAPE 4: Pipeline protégé
            await self._execute_protected_pipeline(audio_chunk, result)
            
            # ✅ ÉTAPE 5: Métriques finales
            total_latency = (time.perf_counter() - pipeline_start) * 1000
            result["latency_ms"] = total_latency
            result["metrics"]["pipeline_latency_ms"] = total_latency
            
            # Mise à jour statistiques
            self._update_performance_stats(total_latency, success=result["success"])
            
            if result["success"]:
                self.metrics.increment_pipeline_requests("success")
                logger.info(f"✅ Pipeline réussi pour {auth_info['username']}: '{result['text'][:50]}...' ({total_latency:.1f}ms)")
            else:
                self.metrics.increment_pipeline_requests("error")
                
        except Exception as e:
            # Gestion d'erreur globale
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg)
            
            result["errors"].append(error_msg)
            result["success"] = False
            
            total_latency = (time.perf_counter() - pipeline_start) * 1000
            result["latency_ms"] = total_latency
            
            self.metrics.increment_pipeline_requests("error")
            self._update_performance_stats(total_latency, success=False)
        
        return result

    # ✅ NOUVEAUTÉ: Pipeline protégé avec circuit breakers
    async def _execute_protected_pipeline(self, audio_chunk: np.ndarray, result: Dict):
        """Exécute le pipeline avec protection complète"""
        
        # Étape VAD protégée
        speech_detected = await self._process_vad_protected(audio_chunk, result)
        
        if not speech_detected:
            result["text"] = ""
            result["confidence"] = 0.0
            result["success"] = True
            return
        
        # Étape STT protégée
        text = await self._process_stt_protected(audio_chunk, result)
        
        if not text:
            result["errors"].append("STT returned empty text")
            return
        
        # Étape LLM protégée (optionnelle)
        enhanced_text = await self._process_llm_protected(text, result)
        
        # Finaliser résultat
        result["text"] = enhanced_text or text
        result["success"] = True
        result["confidence"] = 0.9  # Placeholder
    
    # ✅ NOUVEAUTÉ: VAD protégé
    async def _process_vad_protected(self, audio_chunk: np.ndarray, result: Dict) -> bool:
        """Traite VAD avec circuit breaker"""
        
        if not self.vad_manager:
            result["components_used"]["vad"] = {"status": "disabled", "reason": "not_available"}
            return True  # Pas de VAD, considérer tout comme parole
        
        async def vad_detection():
            vad_start = time.perf_counter()
            
            # Simulation détection (à adapter selon votre implémentation VAD)
            if hasattr(self.vad_manager, 'detect_voice'):
                vad_result = await self.vad_manager.detect_voice(audio_chunk.tobytes())
                speech_detected = vad_result.get('has_voice', False)
                speech_prob = vad_result.get('confidence', 0.0)
            else:
                # Fallback simple basé sur l'énergie
                energy = np.mean(audio_chunk ** 2)
                speech_detected = energy > 0.001
                speech_prob = min(energy * 1000, 1.0)
            
            vad_latency = (time.perf_counter() - vad_start) * 1000
            
            # Métriques
            self.metrics.record_vad_latency(vad_latency / 1000, self.vad_manager.backend)
            
            return {
                "speech_detected": speech_detected,
                "speech_probability": speech_prob,
                "latency_ms": vad_latency,
                "backend": getattr(self.vad_manager, 'backend', 'unknown')
            }
        
        try:
            vad_result = await self.error_handler.execute_safe("vad", vad_detection)
            
            result["components_used"]["vad"] = vad_result
            return vad_result["speech_detected"]
            
        except CircuitOpenError:
            logger.warning("⚠️ Circuit VAD ouvert, fallback")
            result["components_used"]["vad"] = {"status": "circuit_open", "fallback": True}
            return True  # Fallback: considérer comme parole
        except Exception as e:
            logger.warning(f"⚠️ Erreur VAD: {e}")
            result["components_used"]["vad"] = {"status": "error", "error": str(e)}
            return True
    
    # ✅ NOUVEAUTÉ: STT protégé
    async def _process_stt_protected(self, audio_chunk: np.ndarray, result: Dict) -> str:
        """Traite STT avec circuit breaker et retry"""
        
        async def stt_transcription():
            stt_start = time.perf_counter()
            
            # Obtenir modèle STT
            stt_model = self.fallback_manager.get_component("stt")
            if not stt_model:
                raise Exception("Aucun modèle STT disponible")
            
            # Transcription
            text = await self._do_transcribe(stt_model, audio_chunk)
            
            stt_latency = (time.perf_counter() - stt_start) * 1000
            
            # Métriques
            self.metrics.record_stt_latency(stt_latency / 1000)
            
            return {
                "text": text,
                "latency_ms": stt_latency,
                "model": getattr(stt_model, 'model_name', 'unknown')
            }
        
        try:
            stt_result = await self.error_handler.execute_safe("stt", stt_transcription)
            
            result["components_used"]["stt"] = {
                "model": stt_result["model"],
                "latency_ms": stt_result["latency_ms"],
                "status": "success"
            }
            
            return stt_result["text"].strip()
            
        except CircuitOpenError:
            logger.error("❌ Circuit STT ouvert")
            result["components_used"]["stt"] = {"status": "circuit_open"}
            return ""
        except Exception as e:
            logger.error(f"❌ Erreur STT: {e}")
            result["components_used"]["stt"] = {"status": "error", "error": str(e)}
            return ""
    
    # ✅ NOUVEAUTÉ: LLM protégé  
    async def _process_llm_protected(self, text: str, result: Dict) -> Optional[str]:
        """Traite LLM avec circuit breaker (placeholder)"""
        
        async def llm_processing():
            # Simulation traitement LLM
            # Dans une vraie implémentation, ici on ferait:
            # - Correction grammaticale
            # - Amélioration ponctuation
            # - Traitement commandes
            
            await asyncio.sleep(0.1)  # Simulation latence
            return text  # Pour l'instant, retourne le texte tel quel
        
        try:
            enhanced_text = await self.error_handler.execute_safe("llm", llm_processing)
            
            result["components_used"]["llm"] = {
                "processed": True,
                "status": "success"
            }
            
            return enhanced_text
            
        except CircuitOpenError:
            logger.warning("⚠️ Circuit LLM ouvert")
            result["components_used"]["llm"] = {"status": "circuit_open"}
            return None
        except Exception as e:
            logger.warning(f"⚠️ Erreur LLM: {e}")
            result["components_used"]["llm"] = {"status": "error", "error": str(e)}
            return None

    # ✅ NOUVEAUTÉ: Méthode pour obtenir l'état sécurisé
    def get_security_status(self) -> Dict[str, Any]:
        """Retourne l'état de sécurité du système"""
        
        return {
            "timestamp": time.time(),
            "authentication": {
                "api_keys_configured": len(self.security_config.api_keys) > 0,
                "jwt_configured": bool(self.security_config.jwt_secret)
            },
            "circuit_breakers": self.error_handler.get_health_status(),
            "component_protection": {
                name: {
                    "protected": True,
                    "state": cb.state.value,
                    "error_rate": cb.metrics.error_rate
                }
                for name, cb in self.error_handler.circuit_breakers.items()
            }
        }
    
    # ✅ NOUVEAUTÉ: Méthode de diagnostic complète
    def get_health_status(self) -> Dict[str, Any]:
        """Retourne l'état de santé complet du système"""
        
        base_health = {
            "timestamp": time.time(),
            "status": "healthy" if self.is_initialized else "initializing",
            "performance": self.performance_stats.copy(),
            "components": {
                "vad": {
                    "available": self.vad_manager is not None,
                    "backend": getattr(self.vad_manager, 'backend', None)
                },
                "stt": {
                    "preloaded": "stt" in self.components
                },
                "monitoring": {
                    "active": hasattr(self.metrics, 'server_thread')
                }
            },
            "errors": self.error_counts.copy()
        }
        
        # Ajouter état sécurisé
        base_health["security"] = self.get_security_status()
        
        return base_health

    # ✅ BACKWARD COMPATIBILITY: Ancienne méthode maintenue
    async def process_audio_safe(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Version de compatibilité - utilise un token par défaut
        ⚠️ DEPRECATED: Utiliser process_audio_secure() avec authentification
        """
        
        logger.warning("⚠️ Utilisation méthode non sécurisée - Upgrade recommandé")
        
        # Générer token temporaire pour compatibilité
        temp_token = self.security_config.generate_jwt_token({
            "username": "legacy_user",
            "permissions": ["audio_processing"]
        })
        
        return await self.process_audio_secure(
            audio_chunk=audio_chunk,
            jwt_token=temp_token,
            filename="legacy_audio.wav"
        )

    # Test du Master Handler
async def test_master_handler():
    """Test complet du Master Handler"""
    print("🧪 TEST MASTER HANDLER ROBUSTE")
    print("="*40)
    
    # Créer handler
    handler = RobustMasterHandler()
    await handler.initialize()
    
    # Test avec audio synthétique
    print("\n🎯 Test traitement audio...")
    
    # Audio test (3 secondes de bruit)
    test_audio = np.random.randn(48000).astype(np.float32) * 0.1
    
    # Traiter audio
    result = await handler.process_audio_safe(test_audio)
    
    print(f"Résultat: {result}")
    
    # Statut de santé
    print("\n📊 Statut de santé:")
    health = handler.get_health_status()
    print(f"Statut global: {health['status']}")
    print(f"Composants initialisés: {health['initialized']}")
    print(f"Erreurs: {health['performance']['error_counts']}")
    
    print("\n✅ Test Master Handler terminé")

if __name__ == "__main__":
    asyncio.run(test_master_handler())