# STT/stt_manager_robust.py
"""
RobustSTTManager - Gestionnaire STT robuste avec fallback multi-modèles
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
Conforme aux exigences du PRD v3.1 et du Plan de Développement Final
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

print("🎮 GPU Configuration: RTX 3090 (CUDA:0 après mapping)")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"🔒 CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER')}")

# =============================================================================
# 🚨 MEMORY LEAK PREVENTION V4.0 - OBLIGATOIRE 
# =============================================================================
# Import du système de prévention memory leak validé
try:
    from memory_leak_v4 import (
        configure_for_environment, 
        gpu_test_cleanup, 
        validate_no_memory_leak,
        emergency_gpu_reset
    )
    # Configuration environnement (dev/ci/production)
    configure_for_environment("dev")  # Adapter selon contexte
    print("✅ Memory Leak Prevention V4.0 activé")
except ImportError:
    print("⚠️ Memory Leak V4.0 non disponible - Continuer avec validation standard")
    gpu_test_cleanup = lambda name: lambda func: func  # Fallback

# Maintenant imports normaux...
import torch
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import io
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
from prometheus_client import Counter, Histogram, Gauge
from circuitbreaker import circuit
from STT.vad_manager import OptimizedVADManager  # Intégration VAD existant

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métriques Prometheus conformes à l'architecture
stt_transcriptions_total = Counter('stt_transcriptions_total', 'Total transcriptions')
stt_errors_total = Counter('stt_errors_total', 'Total errors')
stt_latency_seconds = Histogram('stt_latency_seconds', 'Transcription latency')
stt_vram_usage_bytes = Gauge('stt_vram_usage_bytes', 'VRAM usage in bytes')

def validate_rtx3090_mandatory():
    """Validation systématique RTX 3090 - OBLIGATOIRE dans chaque script"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    # CONTRÔLE 1: Variables environnement
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER', '')
    
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    if cuda_order != 'PCI_BUS_ID':
        raise RuntimeError(f"🚫 CUDA_DEVICE_ORDER='{cuda_order}' incorrect - doit être 'PCI_BUS_ID'")
    
    # CONTRÔLE 2: GPU physique détecté (après mapping, cuda:0 = RTX 3090)
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 3090" not in gpu_name:
        raise RuntimeError(f"🚫 GPU détecté: {gpu_name} - RTX 3090 requise")
    
    # CONTRÔLE 3: Mémoire GPU
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 ≈ 24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) insuffisante - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")

class RobustSTTManager:
    """
    Manager STT robuste avec:
    - Sélection GPU automatique optimale
    - Chaîne de fallback multi-modèles
    - Gestion VRAM intelligente avec clear_cache
    - Métriques temps réel (latence, erreurs, succès)
    - Conversion audio robuste (bytes ↔ numpy)
    - Intégration VAD existant
    """
    
    def __init__(self, config: Dict[str, Any], vad_manager: Optional[OptimizedVADManager] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validation GPU obligatoire AVANT sélection device
        if config.get('use_gpu', True):
            validate_rtx3090_mandatory()
        
        self.device = self._select_optimal_device()
        self.models = {}
        self.fallback_chain = config.get("fallback_chain", ["base", "tiny"])
        self.vad_manager = vad_manager
        
        # Configuration compute_type selon device
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Métriques internes pour monitoring
        self.metrics = {
            "transcriptions": 0,
            "errors": 0,
            "avg_latency": 0.0,
            "total_latency": 0.0
        }

    def _select_optimal_device(self) -> str:
        """Sélection intelligente du device avec fallback GPU→CPU - RTX 3090 EXCLUSIF"""
        if not self.config.get('use_gpu', True):
            self.logger.info("GPU désactivé par configuration")
            return "cpu"
            
        if torch.cuda.is_available():
            # 🚨 CONFIGURATION CRITIQUE: Après CUDA_VISIBLE_DEVICES='1' et CUDA_DEVICE_ORDER='PCI_BUS_ID'
            # PyTorch voit UNIQUEMENT la RTX 3090 qui devient cuda:0
            # La RTX 5060 Ti est masquée et inaccessible
            
            # Validation obligatoire RTX 3090
            validate_rtx3090_mandatory()
            
            # Utiliser cuda:0 (qui est maintenant RTX 3090 après remapping)
            selected_gpu = 0  # RTX 3090 24GB VRAM après mapping CUDA_VISIBLE_DEVICES
            torch.cuda.set_device(selected_gpu)
            
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(selected_gpu)
            self.logger.info(f"🎮 GPU active: {gpu_name} (Device {selected_gpu}/{gpu_count})")
            
            # Vérification VRAM RTX 3090 (24GB attendus) après mapping
            vram_free = torch.cuda.get_device_properties(selected_gpu).total_memory - torch.cuda.memory_allocated(selected_gpu)
            vram_free_gb = vram_free / (1024**3)
            vram_total_gb = torch.cuda.get_device_properties(selected_gpu).total_memory / (1024**3)
            
            # Vérification suffisante pour STT
            if vram_free_gb < 2.0:
                self.logger.warning(f"VRAM libre insuffisante ({vram_free_gb:.1f}GB < 2GB), fallback CPU")
                return "cpu"
                
            self.logger.info(f"✅ RTX 3090 sélectionnée: {vram_total_gb:.1f}GB totale, {vram_free_gb:.1f}GB libre")
            return "cuda"
        else:
            self.logger.warning("CUDA non disponible, utilisation CPU")
            return "cpu"

    async def initialize(self):
        """Initialisation asynchrone avec chargement modèles"""
        self.logger.info(f"Initialisation RobustSTTManager sur {self.device}")
        
        for model_name in self.fallback_chain:
            try:
                await self._load_model(model_name)
            except Exception as e:
                self.logger.error(f"Échec chargement {model_name}: {e}")
                if model_name == self.fallback_chain[-1]:
                    raise RuntimeError("Aucun modèle STT disponible")

    async def _load_model(self, model_size: str):
        """Chargement optimisé avec faster-whisper"""
        start_time = time.time()
        self.logger.info(f"Chargement modèle '{model_size}' sur '{self.device}'...")
        
        try:
            self.models[model_size] = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=2,  # Parallélisation
                download_root=self.config.get("model_cache_dir", "./models")
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ Modèle {model_size} chargé en {load_time:.2f}s")
            
            # Test rapide du modèle
            test_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
            self.models[model_size].transcribe(test_audio, language="fr")
            
        except Exception as e:
            self.logger.error(f"❌ Échec chargement {model_size}: {e}")
            raise

    @circuit(failure_threshold=3, recovery_timeout=30)
    async def transcribe_audio(self, audio_data: bytes, language: str = "fr") -> Dict[str, Any]:
        """
        Transcription robuste avec circuit breaker et métriques
        
        Args:
            audio_data: Audio en bytes (WAV format)
            language: Code langue (défaut: français)
            
        Returns:
            Dict avec transcription, temps de traitement et device utilisé
        """
        start_time = time.time()
        self.metrics["transcriptions"] += 1
        stt_transcriptions_total.inc()
        
        try:
            # Conversion bytes → numpy avec validation
            audio_array = self._bytes_to_audio_array(audio_data)
            
            # Segmentation VAD si disponible
            if self.vad_manager and self.config.get('use_vad', True):
                self.logger.debug("Segmentation VAD activée")
                speech_segments = await self._process_with_vad(audio_array)
            else:
                speech_segments = [(0, len(audio_array), audio_array)]
            
            # Transcription avec fallback chain
            full_transcription = ""
            segments_info = []
            
            for idx, (start_ms, end_ms, segment) in enumerate(speech_segments):
                segment_text = await self._transcribe_segment(segment, language, idx)
                if segment_text:
                    full_transcription += segment_text + " "
                    segments_info.append({
                        "start": start_ms,
                        "end": end_ms,
                        "text": segment_text
                    })
            
            # Calcul métriques
            processing_time = time.time() - start_time
            self._update_latency_metrics(processing_time)
            stt_latency_seconds.observe(processing_time)
            
            # Monitoring VRAM si GPU - RTX 3090 uniquement
            if self.device == "cuda":
                # Surveiller la VRAM sur la GPU active (RTX 3090 si dual-GPU)
                current_device = torch.cuda.current_device()
                vram_used = torch.cuda.memory_allocated(current_device) / (1024**3)
                stt_vram_usage_bytes.set(vram_used * 1024**3)
                self.logger.debug(f"VRAM utilisée sur GPU {current_device}: {vram_used:.2f}GB")
            
            result = {
                "text": full_transcription.strip(),
                "segments": segments_info,
                "processing_time": processing_time,
                "device": self.device,
                "model_used": self._get_active_model(),
                "metrics": {
                    "audio_duration": len(audio_array) / 16000,
                    "rtf": processing_time / (len(audio_array) / 16000)  # Real-time factor
                }
            }
            
            self.logger.info(f"✅ Transcription réussie en {processing_time:.2f}s (RTF: {result['metrics']['rtf']:.2f})")
            return result
            
        except Exception as e:
            self.metrics["errors"] += 1
            stt_errors_total.inc()
            self.logger.error(f"❌ Erreur transcription: {e}", exc_info=True)
            
            # Clear cache GPU si erreur mémoire
            if "out of memory" in str(e).lower() and self.device == "cuda":
                torch.cuda.empty_cache()
                self.logger.info("Cache GPU vidé après erreur mémoire")
            
            raise

    async def _process_with_vad(self, audio_array: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        """Traitement VAD asynchrone avec timestamps"""
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None, 
            self.vad_manager.process_audio, 
            audio_array
        )
        return segments

    async def _transcribe_segment(self, segment: np.ndarray, language: str, idx: int) -> str:
        """Transcription d'un segment avec fallback chain"""
        for model_name in self.fallback_chain:
            if model_name not in self.models:
                continue
                
            try:
                self.logger.debug(f"Tentative transcription segment {idx} avec {model_name}")
                
                # Transcription asynchrone
                loop = asyncio.get_event_loop()
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self.models[model_name].transcribe(
                        segment,
                        language=language,
                        beam_size=5,
                        best_of=5,
                        temperature=0.0,  # Déterministe
                        condition_on_previous_text=True,
                        vad_filter=False  # VAD déjà appliqué
                    )
                )
                
                text = " ".join([seg.text for seg in segments]).strip()
                if text:  # Succès si texte non vide
                    return text
                    
            except Exception as e:
                self.logger.warning(f"Échec {model_name} sur segment {idx}: {e}")
                continue
                
        self.logger.error(f"Échec complet transcription segment {idx}")
        return ""

    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Conversion robuste bytes → numpy avec validation"""
        try:
            # Tentative 1: soundfile (plus robuste)
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # Conversion mono si nécessaire
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resampling à 16kHz si nécessaire
            if sample_rate != 16000:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
            
            # Normalisation [-1, 1]
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
                
            return audio_array
            
        except Exception as e:
            self.logger.error(f"Erreur conversion audio: {e}")
            raise ValueError(f"Format audio invalide: {e}")

    def _update_latency_metrics(self, latency: float):
        """Mise à jour métriques de latence avec moyenne mobile"""
        self.metrics["total_latency"] += latency
        self.metrics["avg_latency"] = (
            self.metrics["total_latency"] / self.metrics["transcriptions"]
        )
        
        self.logger.debug(
            f"Latence: {latency:.3f}s | "
            f"Moyenne: {self.metrics['avg_latency']:.3f}s | "
            f"Total transcriptions: {self.metrics['transcriptions']}"
        )

    def _get_active_model(self) -> str:
        """Retourne le modèle actuellement chargé"""
        return list(self.models.keys())[0] if self.models else "none"

    def get_metrics(self) -> Dict[str, Any]:
        """Export métriques pour monitoring"""
        return {
            **self.metrics,
            "device": self.device,
            "models_loaded": list(self.models.keys()),
            "vad_enabled": self.vad_manager is not None
        }

    async def cleanup(self):
        """Nettoyage ressources et modèles"""
        self.logger.info("Nettoyage RobustSTTManager...")
        
        # Libération modèles
        self.models.clear()
        
        # Clear cache GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        self.logger.info("✅ Nettoyage terminé")

    def listen_and_transcribe(self, duration: float = 5.0, use_vad: bool = True) -> Dict[str, Any]:
        """
        Version synchrone pour compatibilité avec ancien handler
        Enregistre audio du microphone et transcrit
        """
        import sounddevice as sd
        
        self.logger.info(f"🎤 Enregistrement {duration}s en cours...")
        
        try:
            # Enregistrement
            audio_data = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            # Conversion en bytes WAV
            buffer = io.BytesIO()
            sf.write(buffer, audio_data.flatten(), 16000, format='WAV', subtype='PCM_16')
            audio_bytes = buffer.getvalue()
            
            # Transcription asynchrone
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.transcribe_audio(audio_bytes, "fr"))
            loop.close()
            
            if result['text']:
                self.logger.info(f"✅ Transcription: '{result['text']}'")
                return {
                    "success": True,
                    "transcription": result['text'],
                    "latency_ms": result['processing_time'] * 1000,
                    "model_used": result['model_used'],
                    "device": result['device']
                }
            else:
                return {
                    "success": False,
                    "error": "Transcription vide",
                    "latency_ms": result['processing_time'] * 1000
                }
                
        except Exception as e:
            self.logger.error(f"❌ Erreur enregistrement/transcription: {e}")
            return {
                "success": False,
                "error": str(e),
                "latency_ms": 0
            } 


# APPELER OBLIGATOIREMENT dans __main__ ou au début du script
if __name__ == "__main__":
    print("🧪 VALIDATION RTX 3090 - STT Manager Robust")
    print("=" * 50)
    
    # Validation obligatoire de la configuration
    validate_rtx3090_mandatory()
    
    # Test d'instanciation basique
    test_config = {
        "use_gpu": True,
        "fallback_chain": ["base"],
        "model_cache_dir": "./models"
    }
    
    try:
        manager = RobustSTTManager(test_config)
        print(f"✅ RobustSTTManager initialisé avec succès sur {manager.device}")
        print(f"✅ Modèles configurés: {manager.fallback_chain}")
        print(f"✅ Type compute: {manager.compute_type}")
        print("✅ Configuration GPU RTX 3090 validée avec succès")
        
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        sys.exit(1) 