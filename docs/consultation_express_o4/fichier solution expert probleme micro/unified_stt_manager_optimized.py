#!/usr/bin/env python3
"""
Manager STT Unifié Optimisé - SuperWhisper V6
Architecture complète: Cache → VAD → Backend → Post-processing

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

import asyncio
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
import json
from dataclasses import dataclass, asdict

# Configuration GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ajouter le chemin racine du projet
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports adaptés à la structure SuperWhisper V6
try:
    from STT.backends.prism_stt_backend_optimized import OptimizedPrismSTTBackend
except ImportError:
    # Fallback - utiliser le chemin relatif
    sys.path.append(str(Path(__file__).parent))
    from prism_stt_backend_optimized import OptimizedPrismSTTBackend

try:
    from STT.stt_postprocessor import STTPostProcessor
except ImportError:
    # Fallback - utiliser le chemin relatif
    sys.path.append(str(Path(__file__).parent.parent))
    from stt_postprocessor import STTPostProcessor

# Fallbacks pour modules manquants
try:
    from STT.cache_manager import CacheManager
except ImportError:
    # Fallback - Cache manager simple
    class CacheManager:
        def __init__(self, max_size_mb: int = 200, ttl_seconds: int = 7200):
            self.cache = {}
            self.max_size_mb = max_size_mb
            self.ttl_seconds = ttl_seconds
            self.access_times = {}
        
        async def get(self, key: str):
            """Récupère un élément du cache"""
            if key in self.cache:
                # Vérifier TTL
                if time.time() - self.access_times[key] < self.ttl_seconds:
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.access_times[key]
            return None
        
        async def set(self, key: str, value):
            """Stocke un élément dans le cache"""
            self.cache[key] = value
            self.access_times[key] = time.time()
        
        async def get_stats(self):
            """Statistiques du cache"""
            return {
                "size": len(self.cache),
                "max_size_mb": self.max_size_mb,
                "ttl_seconds": self.ttl_seconds
            }
        
        async def close(self):
            """Ferme le cache"""
            self.cache.clear()
            self.access_times.clear()

try:
    from STT.vad_processor import VADProcessor
except ImportError:
    # Fallback - VAD processor simple
    class VADProcessor:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.chunk_ms = config.get('chunk_ms', 160)
            self.latency_threshold = config.get('latency_threshold', 25)
        
        async def initialize(self):
            """Initialise le VAD"""
            pass
        
        async def process(self, audio: np.ndarray):
            """Traite l'audio avec VAD"""
            # VAD simple - détecte tout comme parole si amplitude > seuil
            threshold = 0.01
            speech_samples = np.where(np.abs(audio) > threshold)[0]
            
            if len(speech_samples) > 0:
                start_sample = speech_samples[0]
                end_sample = speech_samples[-1]
                
                speech_segments = [{
                    'start_time': start_sample / 16000,
                    'end_time': end_sample / 16000,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'confidence': 0.8
                }]
            else:
                speech_segments = []
            
            # Retourner un objet simple avec speech_segments
            class VADResult:
                def __init__(self, segments):
                    self.speech_segments = segments
            
            return VADResult(speech_segments)
        
        def get_statistics(self):
            """Statistiques VAD"""
            return {"processed_chunks": 0, "speech_detected": 0}
        
        async def shutdown(self):
            """Arrêt du VAD"""
            pass

@dataclass
class STTResult:
    """Résultat de transcription enrichi"""
    text: str
    confidence: float
    segments: List[Dict[str, Any]]
    processing_time: float
    device: str
    rtf: float
    backend_used: str
    success: bool
    error: Optional[str] = None
    cache_hit: bool = False
    vad_segments: Optional[List[Dict]] = None
    post_processing_metrics: Optional[Dict] = None

class OptimizedUnifiedSTTManager:
    """Manager STT unifié avec optimisations complètes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Composants
        self.backend = None
        self.post_processor = None
        self.cache_manager = None
        self.vad_processor = None
        
        # Configuration optimisée
        self.model_size = config.get('model', 'large-v2')
        self.compute_type = config.get('compute_type', 'float16')
        
        # Cache optimisé
        self.cache_size_mb = config.get('cache_size_mb', 200)  # Augmenté
        self.cache_ttl = config.get('cache_ttl', 7200)  # 2h
        
        # VAD optimisé
        self.vad_chunk_ms = config.get('vad_chunk_ms', 160)  # Chunks plus petits
        self.vad_latency_threshold = config.get('vad_latency_threshold', 25)  # ms
        
        # Statistiques
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "vad_segments_processed": 0,
            "post_processing_applied": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }
        
        self.initialized = False
    
    def _setup_logging(self) -> logging.Logger:
        """Configuration du logging"""
        logger = logging.getLogger('OptimizedSTTManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def initialize(self):
        """Initialisation asynchrone de tous les composants"""
        try:
            self.logger.info("🚀 Initialisation Manager STT Optimisé...")
            start_time = time.time()
            
            # 1. Cache Manager
            self.logger.info("   📦 Initialisation Cache Manager...")
            self.cache_manager = CacheManager(
                max_size_mb=self.cache_size_mb,
                ttl_seconds=self.cache_ttl
            )
            
            # 2. VAD Processor
            self.logger.info("   🎤 Initialisation VAD Processor...")
            self.vad_processor = VADProcessor({
                'chunk_ms': self.vad_chunk_ms,
                'latency_threshold': self.vad_latency_threshold,
                'model': 'silero_v4'  # VAD Silero optimisé
            })
            await self.vad_processor.initialize()
            
            # 3. Backend STT Optimisé
            self.logger.info("   🧠 Initialisation Backend STT...")
            backend_config = {
                'model': self.model_size,
                'compute_type': self.compute_type,
                'device': 'cuda:1'
            }
            self.backend = OptimizedPrismSTTBackend(backend_config)
            
            # 4. Post-Processor
            self.logger.info("   📝 Initialisation Post-Processor...")
            config_path = self.config.get('post_processor_config')
            self.post_processor = STTPostProcessor(config_path)
            
            init_time = time.time() - start_time
            self.initialized = True
            
            self.logger.info(f"✅ Manager STT Optimisé prêt ({init_time:.2f}s)")
            
            # Test de santé
            await self._health_check()
            
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation: {e}")
            raise
    
    async def _health_check(self):
        """Vérification de santé de tous les composants"""
        try:
            self.logger.info("🔍 Vérification santé des composants...")
            
            # Test backend
            if not self.backend.health_check():
                raise RuntimeError("Backend STT non fonctionnel")
            
            # Test VAD avec audio silencieux
            test_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
            vad_result = await self.vad_processor.process(test_audio)
            
            # Test post-processor
            test_text = "test de santé"
            _, metrics = self.post_processor.process(test_text)
            
            self.logger.info("✅ Tous les composants sont fonctionnels")
            
        except Exception as e:
            self.logger.error(f"❌ Échec vérification santé: {e}")
            raise
    
    async def transcribe(self, audio: np.ndarray, skip_cache: bool = False) -> STTResult:
        """
        Transcription complète avec pipeline optimisé
        
        Args:
            audio: Audio à transcrire (16kHz, float32)
            skip_cache: Ignorer le cache
            
        Returns:
            Résultat de transcription enrichi
        """
        if not self.initialized:
            raise RuntimeError("Manager non initialisé - appelez initialize() d'abord")
        
        start_time = time.perf_counter()
        audio_duration = len(audio) / 16000
        
        try:
            self.stats["total_requests"] += 1
            
            # 1. Vérification cache
            cache_key = None
            if not skip_cache:
                cache_key = self._generate_cache_key(audio)
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    cached_result.cache_hit = True
                    self.logger.info(f"📦 Cache hit pour audio {audio_duration:.1f}s")
                    return cached_result
            
            # 2. VAD - Extraction segments vocaux
            self.logger.info(f"🎤 VAD processing pour audio {audio_duration:.1f}s...")
            vad_result = await self.vad_processor.process(audio)
            
            if not vad_result.speech_segments:
                # Pas de parole détectée
                return STTResult(
                    text="",
                    confidence=0.0,
                    segments=[],
                    processing_time=time.perf_counter() - start_time,
                    device="cuda:1",
                    rtf=0.0,
                    backend_used="vad_no_speech",
                    success=True,
                    vad_segments=[]
                )
            
            self.stats["vad_segments_processed"] += len(vad_result.speech_segments)
            
            # 3. Transcription des segments vocaux
            self.logger.info(f"🧠 Transcription {len(vad_result.speech_segments)} segments...")
            
            if len(vad_result.speech_segments) == 1:
                # Un seul segment - transcription directe
                segment = vad_result.speech_segments[0]
                segment_audio = audio[segment['start_sample']:segment['end_sample']]
                stt_result = await self.backend.transcribe(segment_audio)
            else:
                # Multiples segments - transcription par batch
                stt_result = await self._transcribe_multiple_segments(
                    audio, vad_result.speech_segments
                )
            
            if not stt_result.success:
                raise RuntimeError(f"Échec transcription: {stt_result.error}")
            
            # 4. Post-processing
            self.logger.info("📝 Post-processing...")
            processed_text, post_metrics = self.post_processor.process(
                stt_result.text, stt_result.confidence
            )
            
            if post_metrics["corrections_applied"] > 0:
                self.stats["post_processing_applied"] += 1
                self.logger.info(f"   {post_metrics['corrections_applied']} corrections appliquées")
            
            # 5. Résultat final
            processing_time = time.perf_counter() - start_time
            self.stats["total_processing_time"] += processing_time
            
            final_result = STTResult(
                text=processed_text,
                confidence=min(1.0, stt_result.confidence + post_metrics.get("confidence_boost", 0.0)),
                segments=stt_result.segments,
                processing_time=processing_time,
                device=stt_result.device,
                rtf=processing_time / audio_duration,
                backend_used=stt_result.backend_used,
                success=True,
                cache_hit=False,
                vad_segments=vad_result.speech_segments,
                post_processing_metrics=post_metrics
            )
            
            # 6. Mise en cache
            if not skip_cache and cache_key and len(processed_text) > 5:
                await self.cache_manager.set(cache_key, final_result)
            
            self.logger.info(
                f"✅ Transcription terminée: {processing_time*1000:.0f}ms, "
                f"RTF: {final_result.rtf:.3f}, "
                f"Confiance: {final_result.confidence:.2f}"
            )
            
            return final_result
            
        except Exception as e:
            self.stats["errors"] += 1
            processing_time = time.perf_counter() - start_time
            self.logger.error(f"❌ Erreur transcription: {e}")
            
            return STTResult(
                text="",
                confidence=0.0,
                segments=[],
                processing_time=processing_time,
                device="cuda:1",
                rtf=999.0,
                backend_used="error",
                success=False,
                error=str(e)
            )
    
    async def _transcribe_multiple_segments(self, audio: np.ndarray, 
                                          segments: List[Dict]) -> STTResult:
        """Transcription optimisée de multiples segments"""
        all_segments = []
        all_text_parts = []
        total_confidence = 0.0
        
        for i, segment in enumerate(segments):
            segment_audio = audio[segment['start_sample']:segment['end_sample']]
            
            # Transcription du segment
            result = await self.backend.transcribe(segment_audio)
            
            if result.success and result.text.strip():
                # Ajuster timestamps des segments
                time_offset = segment['start_time']
                for seg in result.segments:
                    seg['start'] += time_offset
                    seg['end'] += time_offset
                
                all_segments.extend(result.segments)
                all_text_parts.append(result.text.strip())
                total_confidence += result.confidence
        
        # Reconstruction du texte complet
        full_text = " ".join(all_text_parts)
        avg_confidence = total_confidence / len(segments) if segments else 0.0
        
        return STTResult(
            text=full_text,
            confidence=avg_confidence,
            segments=all_segments,
            processing_time=0.0,  # Sera calculé par le caller
            device="cuda:1",
            rtf=0.0,
            backend_used=f"multi_segment_{len(segments)}",
            success=True
        )
    
    def _generate_cache_key(self, audio: np.ndarray) -> str:
        """Génère une clé de cache pour l'audio"""
        # Hash basé sur le contenu audio + config
        audio_hash = hash(audio.tobytes())
        config_hash = hash(str(sorted(self.config.items())))
        return f"stt_{audio_hash}_{config_hash}"
    
    async def transcribe_streaming(self, audio_stream, chunk_duration: float = 1.0):
        """
        Transcription en streaming pour temps réel
        
        Args:
            audio_stream: Générateur d'audio chunks
            chunk_duration: Durée des chunks en secondes
        """
        buffer = np.array([], dtype=np.float32)
        chunk_size = int(16000 * chunk_duration)
        
        async for audio_chunk in audio_stream:
            buffer = np.concatenate([buffer, audio_chunk])
            
            # Traiter si buffer suffisant
            if len(buffer) >= chunk_size:
                # Extraire chunk à traiter
                process_chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size//2:]  # Overlap 50%
                
                # Transcription
                result = await self.transcribe(process_chunk, skip_cache=True)
                
                if result.success and result.text.strip():
                    yield result
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Statistiques complètes du manager"""
        base_stats = dict(self.stats)
        
        # Statistiques des composants
        if self.backend:
            base_stats["backend_metrics"] = self.backend.get_metrics()
        
        if self.post_processor:
            base_stats["post_processor_stats"] = self.post_processor.get_statistics()
        
        if self.cache_manager:
            base_stats["cache_stats"] = await self.cache_manager.get_stats()
        
        if self.vad_processor:
            base_stats["vad_stats"] = self.vad_processor.get_statistics()
        
        # Métriques calculées
        if self.stats["total_requests"] > 0:
            base_stats["cache_hit_rate"] = (
                self.stats["cache_hits"] / self.stats["total_requests"] * 100
            )
            base_stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["total_requests"]
            )
            base_stats["error_rate"] = (
                self.stats["errors"] / self.stats["total_requests"] * 100
            )
        
        return base_stats
    
    async def shutdown(self):
        """Arrêt propre de tous les composants"""
        self.logger.info("🛑 Arrêt Manager STT Optimisé...")
        
        try:
            if self.cache_manager:
                await self.cache_manager.close()
            
            if self.vad_processor:
                await self.vad_processor.shutdown()
            
            # Sauvegarder statistiques finales
            final_stats = await self.get_statistics()
            stats_file = f"stt_manager_stats_{int(time.time())}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Statistiques sauvegardées: {stats_file}")
            self.logger.info("✅ Arrêt terminé")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'arrêt: {e}")


# Fonction utilitaire pour usage simple
async def create_optimized_stt_manager(config: Dict[str, Any]) -> OptimizedUnifiedSTTManager:
    """
    Crée et initialise un manager STT optimisé
    
    Args:
        config: Configuration du manager
        
    Returns:
        Manager initialisé et prêt
    """
    manager = OptimizedUnifiedSTTManager(config)
    await manager.initialize()
    return manager


if __name__ == "__main__":
    # Test du manager
    async def test_manager():
        config = {
            'model': 'large-v2',
            'compute_type': 'float16',
            'cache_size_mb': 100,
            'cache_ttl': 3600
        }
        
        manager = await create_optimized_stt_manager(config)
        
        # Test avec audio silencieux
        test_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1  # 3s
        result = await manager.transcribe(test_audio)
        
        print(f"Résultat: {result.text}")
        print(f"Confiance: {result.confidence:.2f}")
        print(f"Temps: {result.processing_time*1000:.0f}ms")
        
        # Statistiques
        stats = await manager.get_statistics()
        print(f"Stats: {stats}")
        
        await manager.shutdown()
    
    asyncio.run(test_manager()) 