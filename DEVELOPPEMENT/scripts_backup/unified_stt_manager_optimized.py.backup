#!/usr/bin/env python3
"""
Manager STT Unifié Optimisé - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
Architecture complète: Cache → VAD → Backend → Post-processing
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import asyncio
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import des composants optimisés
try:
    from STT.backends.prism_stt_backend_optimized import OptimizedPrismSTTBackend
    from STT.stt_postprocessor import STTPostProcessor
except ImportError:
    # Fallback pour imports relatifs
    import sys
    sys.path.append(str(Path(__file__).parent))
    from backends.prism_stt_backend_optimized import OptimizedPrismSTTBackend
    from stt_postprocessor import STTPostProcessor

class OptimizedUnifiedSTTManager:
    """Manager STT unifié avec optimisations complètes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Composants
        self.backend = None
        self.post_processor = None
        
        # Configuration optimisée
        self.model_size = config.get('model', 'large-v2')
        self.compute_type = config.get('compute_type', 'float16')
        
        # Statistiques détaillées
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "total_audio_duration": 0.0,
            "post_processing_applied": 0,
            "average_rtf": 0.0,
            "average_confidence": 0.0,
            "total_corrections": 0,
            "initialization_time": 0.0
        }
        
        self.initialized = False
        self.start_time = time.time()
    
    def validate_rtx3090_configuration(self):
        """Validation obligatoire de la configuration RTX 3090"""
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices != '1':
            raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        gpu_name = torch.cuda.get_device_name(0)
        self.logger.info(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")
    
    async def initialize(self):
        """Initialisation asynchrone de tous les composants"""
        try:
            self.logger.info("🚀 Initialisation Manager STT Optimisé...")
            start_time = time.time()
            
            # Validation GPU RTX 3090
            self.validate_rtx3090_configuration()
            
            # Backend STT Optimisé
            self.logger.info("   🧠 Initialisation Backend STT Optimisé...")
            backend_config = {
                'model': self.model_size,
                'compute_type': self.compute_type,
                'device': 'cuda:1'
            }
            self.backend = OptimizedPrismSTTBackend(backend_config)
            
            # Post-Processor
            self.logger.info("   📝 Initialisation Post-Processor...")
            post_config_path = self.config.get('post_processor_config')
            self.post_processor = STTPostProcessor(post_config_path)
            
            init_time = time.time() - start_time
            self.stats["initialization_time"] = init_time
            self.initialized = True
            
            self.logger.info(f"✅ Manager STT Optimisé prêt ({init_time:.2f}s)")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation: {e}")
            raise
    
    async def transcribe(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcription complète avec pipeline optimisé"""
        if not self.initialized:
            raise RuntimeError("Manager non initialisé - appelez initialize() d'abord")
        
        start_time = time.perf_counter()
        audio_duration = len(audio) / 16000
        
        try:
            self.stats["total_requests"] += 1
            self.stats["total_audio_duration"] += audio_duration
            
            self.logger.info(f"🧠 Transcription audio {audio_duration:.1f}s (modèle: {self.model_size})...")
            
            # 1. Validation audio
            audio = self._validate_audio(audio)
            
            # 2. Transcription avec backend optimisé
            stt_result = await self.backend.transcribe(audio)
            
            if not stt_result['success']:
                raise RuntimeError(f"Échec transcription: {stt_result.get('error')}")
            
            # 3. Post-processing
            self.logger.info("📝 Post-processing...")
            processed_text, post_metrics = self.post_processor.process(
                stt_result['text'], stt_result['confidence']
            )
            
            if post_metrics["corrections_applied"] > 0:
                self.stats["post_processing_applied"] += 1
                self.stats["total_corrections"] += post_metrics["corrections_applied"]
                self.logger.info(f"   {post_metrics['corrections_applied']} corrections appliquées")
            
            # 4. Calcul métriques finales
            processing_time = time.perf_counter() - start_time
            rtf = processing_time / audio_duration
            final_confidence = min(1.0, stt_result['confidence'] + post_metrics.get("confidence_boost", 0.0))
            
            # 5. Mise à jour statistiques
            self.stats["total_processing_time"] += processing_time
            self.stats["successful_requests"] += 1
            self._update_running_averages(rtf, final_confidence)
            
            # 6. Résultat final
            final_result = {
                'text': processed_text,
                'confidence': final_confidence,
                'segments': stt_result['segments'],
                'processing_time': processing_time,
                'rtf': rtf,
                'audio_duration': audio_duration,
                'success': True,
                'post_processing_metrics': post_metrics,
                'model_used': self.model_size,
                'backend_metrics': {
                    'backend_processing_time': stt_result['processing_time'],
                    'backend_rtf': stt_result['rtf']
                }
            }
            
            self.logger.info(
                f"✅ Transcription terminée: {processing_time*1000:.0f}ms, "
                f"RTF: {rtf:.3f}, "
                f"Confiance: {final_confidence:.2f}"
            )
            
            return final_result
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self.stats["failed_requests"] += 1
            self.logger.error(f"❌ Erreur transcription: {e}")
            
            return {
                'text': "",
                'confidence': 0.0,
                'segments': [],
                'processing_time': processing_time,
                'rtf': 999.0,
                'audio_duration': audio_duration,
                'success': False,
                'error': str(e),
                'model_used': self.model_size
            }
    
    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """Validation et normalisation de l'audio"""
        if audio is None or len(audio) == 0:
            raise ValueError("Audio vide ou None")
        
        # Conversion en float32 si nécessaire
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalisation si nécessaire
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
            self.logger.warning("⚠️ Audio normalisé (dépassement [-1,1])")
        
        # Vérification durée minimale
        min_duration = 0.1  # 100ms minimum
        if len(audio) / 16000 < min_duration:
            self.logger.warning(f"⚠️ Audio très court: {len(audio)/16000:.3f}s")
        
        return audio
    
    def _update_running_averages(self, rtf: float, confidence: float):
        """Met à jour les moyennes courantes"""
        n = self.stats["successful_requests"]
        
        # Moyenne RTF
        current_avg_rtf = self.stats["average_rtf"]
        self.stats["average_rtf"] = (current_avg_rtf * (n-1) + rtf) / n
        
        # Moyenne confiance
        current_avg_conf = self.stats["average_confidence"]
        self.stats["average_confidence"] = (current_avg_conf * (n-1) + confidence) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Statistiques complètes du manager"""
        stats = dict(self.stats)
        
        # Calculs dérivés
        if self.stats["total_requests"] > 0:
            stats["success_rate"] = (
                self.stats["successful_requests"] / self.stats["total_requests"] * 100
            )
            stats["failure_rate"] = (
                self.stats["failed_requests"] / self.stats["total_requests"] * 100
            )
            
        if self.stats["successful_requests"] > 0:
            stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["successful_requests"]
            )
            
        if self.stats["total_audio_duration"] > 0:
            stats["overall_rtf"] = (
                self.stats["total_processing_time"] / self.stats["total_audio_duration"]
            )
            
        # Métriques post-processing
        if self.stats["total_requests"] > 0:
            stats["post_processing_rate"] = (
                self.stats["post_processing_applied"] / self.stats["total_requests"] * 100
            )
            stats["avg_corrections_per_request"] = (
                self.stats["total_corrections"] / self.stats["total_requests"]
            )
        
        # Temps de fonctionnement
        stats["uptime_seconds"] = time.time() - self.start_time
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """État de santé du système"""
        stats = self.get_statistics()
        
        health = {
            "status": "healthy",
            "issues": [],
            "performance": "good"
        }
        
        # Vérifications santé
        if stats.get("failure_rate", 0) > 10:
            health["status"] = "degraded"
            health["issues"].append(f"Taux d'échec élevé: {stats['failure_rate']:.1f}%")
        
        if stats.get("average_rtf", 0) > 0.5:
            health["performance"] = "slow"
            health["issues"].append(f"RTF élevé: {stats['average_rtf']:.3f}")
        
        if stats.get("average_confidence", 1) < 0.7:
            health["performance"] = "poor"
            health["issues"].append(f"Confiance faible: {stats['average_confidence']:.2f}")
        
        if not self.initialized:
            health["status"] = "not_ready"
            health["issues"].append("Manager non initialisé")
        
        return health
    
    def reset_statistics(self):
        """Remet à zéro les statistiques"""
        init_time = self.stats["initialization_time"]
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "total_audio_duration": 0.0,
            "post_processing_applied": 0,
            "average_rtf": 0.0,
            "average_confidence": 0.0,
            "total_corrections": 0,
            "initialization_time": init_time
        }
        self.start_time = time.time()
        self.logger.info("📊 Statistiques réinitialisées")
    
    def _setup_logging(self):
        logger = logging.getLogger('OptimizedSTTManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# Test rapide du manager
if __name__ == "__main__":
    async def test_manager():
        """Test rapide du manager optimisé"""
        print("🧪 Test Manager STT Optimisé")
        print("=" * 50)
        
        # Configuration test
        config = {
            'model': 'large-v2',
            'compute_type': 'float16'
        }
        
        # Initialisation
        manager = OptimizedUnifiedSTTManager(config)
        await manager.initialize()
        
        # Audio test simulé
        duration = 3.0  # 3 secondes
        samples = int(16000 * duration)
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        
        print(f"🎵 Test avec audio {duration}s ({samples} échantillons)")
        
        # Test transcription
        result = await manager.transcribe(audio)
        
        if result['success']:
            print(f"✅ Transcription réussie:")
            print(f"   Texte: '{result['text']}'")
            print(f"   Confiance: {result['confidence']:.2f}")
            print(f"   RTF: {result['rtf']:.3f}")
            print(f"   Temps: {result['processing_time']*1000:.0f}ms")
        else:
            print(f"❌ Échec: {result.get('error')}")
        
        # Statistiques
        print(f"\n📊 Statistiques:")
        stats = manager.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        # Santé
        health = manager.get_health_status()
        print(f"\n🏥 Santé: {health['status']} ({health['performance']})")
        if health['issues']:
            for issue in health['issues']:
                print(f"   ⚠️ {issue}")
    
    # Exécution test
    asyncio.run(test_manager()) 