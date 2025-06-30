#!/usr/bin/env python3
"""
Backend STT utilisant Prism_Whisper2 - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Intégration du projet Prism_Whisper2 optimisé pour RTX 3090
Performance cible: 4.5s → < 400ms avec optimisations SuperWhisper V6

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
import asyncio
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 SuperWhisper V6 STT - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Imports après configuration GPU
try:
    import torch
    from faster_whisper import WhisperModel
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    print("Installation requise: pip install faster-whisper torch")
    sys.exit(1)

# Import modules locaux
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from STT.backends.base_stt_backend import BaseSTTBackend, STTResult, validate_rtx3090_mandatory
from STT.model_pool import model_pool

class PrismSTTBackend(BaseSTTBackend):
    """
    Backend STT Prism_Whisper2 optimisé RTX 3090 - SuperWhisper V6
    
    Basé sur l'analyse de Prism_Whisper2 avec optimisations SuperWhisper V6:
    - faster-whisper avec compute_type="float16" 
    - GPU Memory Optimizer intégré
    - Cache modèles intelligent
    - Performance cible < 400ms pour 5s audio
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le backend Prism STT
        
        Args:
            config: Configuration avec model_size, compute_type, etc.
        """
        super().__init__(config)
        
        # Configuration Prism
        self.model_size = config.get('model', 'large-v2')
        self.compute_type = config.get('compute_type', 'float16')
        self.language = config.get('language', 'fr')
        self.beam_size = config.get('beam_size', 5)
        self.vad_filter = config.get('vad_filter', True)  # 🔧 VAD avec paramètres corrigés pour transcription complète
        
        # Modèle Whisper
        self.model = None
        self.model_loaded = False
        
        # Optimisations mémoire (inspiré Prism_Whisper2)
        self.memory_optimizer = None
        self.pinned_buffers = []
        
        # Métriques spécifiques Prism
        self.model_load_time = 0.0
        self.warm_up_completed = False
        
        self.logger = self._setup_logging()
        
        # Initialisation
        self._initialize_prism_backend()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging pour Prism backend"""
        logger = logging.getLogger(f'PrismSTTBackend_{self.model_size}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - Prism - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_prism_backend(self):
        """Initialise le backend Prism avec optimisations RTX 3090"""
        try:
            self.logger.info(f"🚀 Initialisation Prism STT {self.model_size} sur RTX 3090...")
            
            # Validation GPU obligatoire
            validate_rtx3090_mandatory()
            
            # Chargement du modèle depuis le pool partagé
            start_time = time.time()
            self.model = model_pool.get_model(self.model_size, self.compute_type)
            
            if self.model is None:
                raise RuntimeError(f"Impossible de charger le modèle '{self.model_size}' depuis le pool.")

            self.model_load_time = time.time() - start_time
            self.model_loaded = True
            
            self.logger.info(f"✅ Modèle '{self.model_size}' obtenu depuis le pool en {self.model_load_time:.2f}s")
            
            # Warm-up GPU avec audio test (inspiré Prism_Whisper2)
            self._warm_up_model()
            
            # Initialiser optimiseur mémoire
            self._initialize_memory_optimizer()
            
            self.logger.info("🎤 Backend Prism STT prêt sur RTX 3090")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation Prism: {e}")
            raise RuntimeError(f"Échec initialisation PrismSTTBackend: {e}")
    
    def _warm_up_model(self):
        """Warm-up modèle avec audio test (inspiré Prism_Whisper2)"""
        try:
            self.logger.info("🔥 Warm-up modèle Prism...")
            
            # Audio test 3 secondes (comme dans Prism_Whisper2)
            dummy_audio = np.zeros(16000 * 3, dtype=np.float32)
            
            # 3 passes de warm-up
            for i in range(3):
                start_time = time.time()
                segments, _ = self.model.transcribe(
                    dummy_audio,
                    language=self.language,
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter
                )
                # Consommer les segments pour forcer l'exécution
                list(segments)
                
                warm_up_time = time.time() - start_time
                self.logger.info(f"   Warm-up {i+1}/3: {warm_up_time:.3f}s")
            
            self.warm_up_completed = True
            self.logger.info("✅ Warm-up Prism terminé")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Warm-up échoué: {e}")
    
    def _initialize_memory_optimizer(self):
        """Initialise optimiseur mémoire (inspiré Prism_Whisper2)"""
        try:
            # Pré-allocation buffers pinned pour audio
            buffer_sizes = [16000 * 1, 16000 * 3, 16000 * 5, 16000 * 10]  # 1s, 3s, 5s, 10s
            
            for size in buffer_sizes:
                buffer = torch.zeros(size, dtype=torch.float32, pin_memory=True)
                self.pinned_buffers.append({
                    'size': size,
                    'buffer': buffer,
                    'in_use': False
                })
            
            self.logger.info(f"📦 {len(self.pinned_buffers)} buffers pinned pré-alloués")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Optimiseur mémoire: {e}")
    
    def _get_optimal_buffer(self, audio_size: int) -> Optional[torch.Tensor]:
        """Obtient buffer pinned optimal pour taille audio"""
        best_buffer = None
        best_size_diff = float('inf')
        
        for buffer_info in self.pinned_buffers:
            if (not buffer_info['in_use'] and 
                buffer_info['size'] >= audio_size):
                
                size_diff = buffer_info['size'] - audio_size
                if size_diff < best_size_diff:
                    best_buffer = buffer_info
                    best_size_diff = size_diff
        
        if best_buffer:
            best_buffer['in_use'] = True
            return best_buffer['buffer'][:audio_size]
        
        return None
    
    def _release_buffer(self, buffer: torch.Tensor):
        """Libère buffer pinned"""
        for buffer_info in self.pinned_buffers:
            if torch.equal(buffer_info['buffer'][:len(buffer)], buffer):
                buffer_info['in_use'] = False
                break
    
    async def transcribe(self, audio: np.ndarray) -> STTResult:
        """
        Transcription asynchrone avec optimisations Prism_Whisper2
        
        Args:
            audio: Audio 16kHz mono float32
            
        Returns:
            STTResult avec transcription et métriques
        """
        if not self.model_loaded:
            raise RuntimeError("Modèle Prism non chargé")
        
        start_time = time.perf_counter()
        audio_duration = len(audio) / 16000  # secondes
        
        try:
            # Optimisation mémoire avec buffer pinned
            gpu_buffer = self._get_optimal_buffer(len(audio))
            if gpu_buffer is not None:
                # Copie optimisée vers buffer pinned
                gpu_buffer.copy_(torch.from_numpy(audio))
                audio_for_transcription = gpu_buffer.cpu().numpy()
            else:
                # Fallback copie standard
                audio_for_transcription = audio.copy()
            
            # Transcription dans thread séparé (éviter blocage asyncio)
            result = await asyncio.to_thread(
                self._transcribe_sync,
                audio_for_transcription
            )
            
            # Libérer buffer si utilisé
            if gpu_buffer is not None:
                self._release_buffer(gpu_buffer)
            
            processing_time = time.perf_counter() - start_time
            rtf = processing_time / audio_duration
            
            # Enregistrer métriques
            self._record_request(processing_time, True)
            
            return STTResult(
                text=result['text'],
                confidence=result['confidence'],
                segments=result['segments'],
                processing_time=processing_time,
                device=self.device,
                rtf=rtf,
                backend_used=f"prism_{self.model_size}",
                success=True
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self._record_request(processing_time, False)
            
            self.logger.error(f"❌ Erreur transcription Prism: {e}")
            
            return STTResult(
                text="",
                confidence=0.0,
                segments=[],
                processing_time=processing_time,
                device=self.device,
                rtf=999.0,
                backend_used=f"prism_{self.model_size}",
                success=False,
                error=str(e)
            )
    
    def _transcribe_sync(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Transcription synchrone pour thread - optimisée Prism_Whisper2
        🔧 VAD CORRIGÉ: Paramètres EXPERTS faster-whisper pour transcription complète
        
        Args:
            audio: Audio numpy array
            
        Returns:
            Dict avec text, confidence, segments
        """
        try:
            # ✅ PARAMÈTRES VAD CORRECTS pour faster-whisper (SOLUTION EXPERTE)
            vad_parameters = {
                "threshold": 0.3,                    # Plus permissif (défaut: 0.5)
                "min_speech_duration_ms": 100,       # Détection plus rapide (défaut: 250)
                "max_speech_duration_s": float('inf'), # Pas de limite (défaut: 30s)
                "min_silence_duration_ms": 2000,     # 2s de silence pour couper (défaut: 2000)
                "speech_pad_ms": 400                 # Padding autour de la parole (défaut: 400)
            }
            
            # Transcription avec paramètres VAD corrects
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=self.beam_size,
                best_of=5,
                vad_filter=self.vad_filter,
                vad_parameters=vad_parameters if self.vad_filter else None,
                word_timestamps=False,
                condition_on_previous_text=True,  # Améliore la cohérence
                without_timestamps=False,          # Garde les timestamps
                initial_prompt=None,               # Pas de prompt initial
                temperature=0.0,                   # Déterministe
                compression_ratio_threshold=2.4,   # Standard
                log_prob_threshold=-1.0,          # Standard
                no_speech_threshold=0.6,          # Standard
                prepend_punctuations="\"'¿([{-",
                append_punctuations="\"'.。,，!！?？:：\")]}、"
            )
            
            # Extraire texte et segments
            text_parts = []
            segments_list = []
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                text_parts.append(segment.text)
                segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'confidence': getattr(segment, 'avg_logprob', 0.8)  # Fallback confidence
                })
                
                # Calculer confiance moyenne
                if hasattr(segment, 'avg_logprob'):
                    # Convertir log prob en confidence (approximation)
                    confidence = min(1.0, max(0.0, (segment.avg_logprob + 1.0)))
                    total_confidence += confidence
                    segment_count += 1
            
            # Texte final
            final_text = ' '.join(text_parts).strip()
            
            # Confiance moyenne
            avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.8
            
            return {
                'text': final_text,
                'confidence': avg_confidence,
                'segments': segments_list,
                'language': info.language if hasattr(info, 'language') else self.language,
                'language_probability': getattr(info, 'language_probability', 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur transcription sync: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Vérifie l'état de santé du backend Prism
        
        Returns:
            True si le backend est opérationnel
        """
        try:
            if not self.model_loaded:
                return False
            
            # Test rapide avec audio court
            test_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
            
            start_time = time.time()
            segments, _ = self.model.transcribe(test_audio, language=self.language)
            list(segments)  # Consommer
            
            health_check_time = time.time() - start_time
            
            # Santé OK si < 2s pour 1s audio
            is_healthy = health_check_time < 2.0
            
            if is_healthy:
                self.logger.debug(f"✅ Health check OK: {health_check_time:.3f}s")
            else:
                self.logger.warning(f"⚠️ Health check lent: {health_check_time:.3f}s")
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"❌ Health check échoué: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne métriques spécifiques Prism"""
        base_metrics = super().get_metrics()
        
        prism_metrics = {
            "model_size": self.model_size,
            "compute_type": self.compute_type,
            "model_load_time": self.model_load_time,
            "warm_up_completed": self.warm_up_completed,
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "pinned_buffers_count": len(self.pinned_buffers),
            "pinned_buffers_in_use": sum(1 for b in self.pinned_buffers if b['in_use'])
        }
        
        # Fusionner métriques
        base_metrics.update(prism_metrics)
        return base_metrics
    
    def cleanup(self):
        """Nettoyage ressources Prism"""
        try:
            # Libérer buffers pinned
            for buffer_info in self.pinned_buffers:
                buffer_info['in_use'] = False
            
            # Nettoyer cache GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("🧹 Nettoyage Prism terminé")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur nettoyage: {e}")
    
    def __del__(self):
        """Destructeur avec nettoyage"""
        self.cleanup() 