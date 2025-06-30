#!/usr/bin/env python3
"""
VAD Optimized Manager - Luxa v1.1 Enhanced
===========================================
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Gestionnaire VAD avancé avec context management, fallbacks intelligents et optimisations temps réel.
Extension du OptimizedVADManager existant avec nouvelles fonctionnalités pour la Tâche 4.

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

import numpy as np
import time
import torch
import asyncio
import logging
from typing import Optional, Tuple, Dict, List, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

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

# Import du VAD Manager existant comme base
try:
    from .vad_manager import OptimizedVADManager
except ImportError:
    # Import direct pour test standalone
    from vad_manager import OptimizedVADManager

@dataclass
class SpeechSegment:
    """Représente un segment de parole détecté"""
    start_time: float
    end_time: float
    probability: float
    chunk_count: int
    energy_level: float

@dataclass
class ConversationContext:
    """Contexte conversationnel pour optimiser la détection"""
    speaker_patterns: Dict[str, float]  # Patterns de parole du locuteur
    silence_patterns: List[float]       # Durées de silence typiques
    energy_baseline: float              # Niveau d'énergie de base
    last_speech_time: Optional[float]   # Timestamp dernière parole
    conversation_duration: float        # Durée totale conversation

class VADOptimizedManager(OptimizedVADManager):
    """
    VAD Manager optimisé avec context management et fallbacks avancés
    
    Nouvelles fonctionnalités par rapport à OptimizedVADManager:
    - Context management conversationnel
    - Cache intelligent des segments récurrents
    - Adaptation dynamique des seuils
    - Historique des performances
    - Optimisations GPU RTX 3090 exclusives
    """
    
    def __init__(self, 
                 chunk_ms: int = 32,   # Adapté pour Silero (512 samples @ 16kHz)
                 latency_threshold_ms: float = 15,  # Seuil plus strict
                 context_window_size: int = 50,     # Fenêtre de contexte
                 adaptive_thresholds: bool = True,  # Seuils adaptatifs
                 enable_caching: bool = True):      # Cache intelligent
        
        # Validation RTX 3090 obligatoire
        validate_rtx3090_mandatory()
        
        # Initialiser la classe parente
        super().__init__(chunk_ms, latency_threshold_ms)
        
        # Configuration avancée
        self.context_window_size = context_window_size
        self.adaptive_thresholds = adaptive_thresholds
        self.enable_caching = enable_caching
        
        # Context Management
        self.conversation_context = ConversationContext(
            speaker_patterns={},
            silence_patterns=[],
            energy_baseline=0.0,
            last_speech_time=None,
            conversation_duration=0.0
        )
        
        # Historique et cache
        self.detection_history = deque(maxlen=context_window_size)
        self.energy_history = deque(maxlen=context_window_size)
        self.segment_cache = {}  # Cache des segments fréquents
        
        # Seuils adaptatifs
        self.current_speech_threshold = 0.5
        self.current_energy_threshold = 0.001
        self.threshold_adaptation_rate = 0.01
        
        # Métriques avancées
        self.advanced_metrics = {
            "context_hits": 0,
            "cache_hits": 0,
            "threshold_adaptations": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
        
        # Thread pool pour traitement asynchrone
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        
        logging.info(f"🚀 VAD Optimized Manager RTX 3090 initialisé:")
        logging.info(f"   Fenêtre contexte: {context_window_size} chunks")
        logging.info(f"   Seuils adaptatifs: {adaptive_thresholds}")
        logging.info(f"   Cache intelligent: {enable_caching}")
        logging.info(f"   🎮 GPU CONFIG: RTX 3090 exclusif via CUDA_VISIBLE_DEVICES='1'")

    async def initialize_optimized(self):
        """Initialisation optimisée avec pré-chargement"""
        # Initialiser la base
        await super().initialize()
        
        # Optimisations spécifiques RTX 3090
        if self.backend == "silero" and torch.cuda.is_available():
            await self._optimize_gpu_memory()
        
        # Pré-charger le cache avec patterns communs
        if self.enable_caching:
            await self._preload_common_patterns()
        
        logging.info("✅ VAD Optimized Manager RTX 3090 initialisé avec succès")

    async def _optimize_gpu_memory(self):
        """Optimisations GPU spécifiques - RTX 3090 UNIQUEMENT"""
        try:
            # RTX 3090 seule visible = device 0 automatiquement
            target_device = 'cuda'  # RTX 3090 automatiquement (seule visible)
            
            # Vérifier que la RTX 3090 est disponible
            if not torch.cuda.is_available():
                logging.warning("⚠️ RTX 3090 non disponible - fallback CPU")
                return
                
            # RTX 3090 est automatiquement CUDA:0 (seule visible)
            # Préallocation mémoire GPU sur RTX 3090
            if hasattr(self.vad_model, 'to'):
                self.vad_model = self.vad_model.to(target_device).half()  # FP16 pour économiser VRAM
            
            # Warmup optimisé sur RTX 3090
            dummy_tensor = torch.randn(self.chunk_samples, dtype=torch.float16, device=target_device)
            with torch.no_grad():
                for _ in range(10):
                    _ = self.vad_model(dummy_tensor, 16000)
            
            torch.cuda.empty_cache()
            logging.info(f"🔧 Optimisations GPU appliquées sur RTX 3090 ({target_device})")
            
        except Exception as e:
            logging.warning(f"⚠️ Optimisations GPU RTX 3090 échouées: {e}")

    async def _preload_common_patterns(self):
        """Pré-charger des patterns audio communs dans le cache"""
        try:
            # Skip si backend n'est pas fonctionnel
            if self.backend == "none":
                logging.info("📦 Cache désactivé (backend pass-through)")
                return
                
            # Générer des patterns de test pour initialiser le cache
            common_patterns = [
                np.zeros(self.chunk_samples, dtype=np.float32),  # Silence
                np.random.randn(self.chunk_samples).astype(np.float32) * 0.1,  # Bruit faible
                np.random.randn(self.chunk_samples).astype(np.float32) * 0.5,  # Signal fort
            ]
            
            for i, pattern in enumerate(common_patterns):
                pattern_hash = hash(pattern.tobytes())
                result = await self._detect_speech_base(pattern)
                self.segment_cache[pattern_hash] = {
                    'result': result,
                    'timestamp': time.time(),
                    'usage_count': 0
                }
            
            logging.info(f"📦 Cache RTX 3090 initialisé avec {len(common_patterns)} patterns")
            
        except Exception as e:
            logging.warning(f"⚠️ Échec initialisation cache RTX 3090: {e}")

    async def detect_speech_optimized(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Détection de parole optimisée avec context management sur RTX 3090
        
        Returns:
            Dict contenant:
            - is_speech: bool
            - probability: float
            - energy_level: float
            - context_confidence: float
            - processing_time_ms: float
            - source: str (cache/context/direct)
        """
        start_time = time.perf_counter()
        
        try:
            # Calcul niveau d'énergie
            energy_level = float(np.mean(audio_chunk ** 2))
            
            # 1. Vérifier le cache
            if self.enable_caching:
                cached_result = await self._check_cache(audio_chunk)
                if cached_result:
                    cached_result['processing_time_ms'] = (time.perf_counter() - start_time) * 1000
                    self.advanced_metrics["cache_hits"] += 1
                    return cached_result
            
            # 2. Détection avec contexte
            result = await self._detect_with_context(audio_chunk, energy_level)
            
            # 3. Mise à jour cache
            if self.enable_caching:
                await self._update_cache(audio_chunk, result)
            
            # 4. Mise à jour contexte
            await self._update_context(result, energy_level)
            
            # 5. Adaptation des seuils
            if self.adaptive_thresholds:
                await self._adapt_thresholds(result, energy_level)
            
            result['processing_time_ms'] = (time.perf_counter() - start_time) * 1000
            return result
            
        except Exception as e:
            logging.error(f"❌ Erreur détection VAD RTX 3090: {e}")
            return {
                'is_speech': False,
                'probability': 0.0,
                'energy_level': 0.0,
                'context_confidence': 0.0,
                'processing_time_ms': (time.perf_counter() - start_time) * 1000,
                'source': 'error',
                'error': str(e)
            }

    async def _check_cache(self, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """Vérifier si le chunk audio est dans le cache"""
        try:
            chunk_hash = hash(audio_chunk.tobytes())
            
            if chunk_hash in self.segment_cache:
                cached_entry = self.segment_cache[chunk_hash]
                
                # Vérifier si le cache n'est pas trop ancien (5 minutes)
                if time.time() - cached_entry['timestamp'] < 300:
                    cached_entry['usage_count'] += 1
                    result = cached_entry['result'].copy()
                    result['source'] = 'cache'
                    result['cache_age_s'] = time.time() - cached_entry['timestamp']
                    return result
                else:
                    # Supprimer entrée expirée
                    del self.segment_cache[chunk_hash]
            
            return None
            
        except Exception as e:
            logging.warning(f"⚠️ Erreur vérification cache: {e}")
            return None

    async def _detect_with_context(self, audio_chunk: np.ndarray, energy_level: float) -> Dict[str, Any]:
        """Détection avec prise en compte du contexte conversationnel"""
        
        # Détection de base
        base_result = await self._detect_speech_base(audio_chunk)
        
        # Calcul de la confiance contextuelle
        context_confidence = 1.0
        
        # Facteur basé sur l'historique récent
        if len(self.detection_history) > 5:
            recent_speech_rate = sum(1 for r in list(self.detection_history)[-10:] if r.get('is_speech', False)) / min(10, len(self.detection_history))
            if recent_speech_rate > 0.7:  # Conversation active
                context_confidence *= 1.2
            elif recent_speech_rate < 0.1:  # Silence prolongé
                context_confidence *= 0.8
        
        # Facteur basé sur l'énergie
        if len(self.energy_history) > 0:
            avg_energy = np.mean(list(self.energy_history))
            if energy_level > avg_energy * 2:  # Énergie inhabituelle
                context_confidence *= 1.1
            elif energy_level < avg_energy * 0.5:  # Énergie faible
                context_confidence *= 0.9
        
        # Facteur temporel (depuis dernière parole)
        if self.conversation_context.last_speech_time:
            time_since_speech = time.time() - self.conversation_context.last_speech_time
            if time_since_speech < 2.0:  # Continuation probable
                context_confidence *= 1.15
            elif time_since_speech > 10.0:  # Pause longue
                context_confidence *= 0.85
        
        # Application de la confiance contextuelle
        adjusted_probability = base_result.get('probability', 0.0) * context_confidence
        adjusted_is_speech = adjusted_probability > self.current_speech_threshold
        
        return {
            'is_speech': adjusted_is_speech,
            'probability': adjusted_probability,
            'base_probability': base_result.get('probability', 0.0),
            'energy_level': energy_level,
            'context_confidence': context_confidence,
            'source': 'context'
        }

    async def _detect_speech_base(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Détection de parole de base (délègue au VAD principal)"""
        try:
            # Utiliser la méthode du parent
            is_speech = await super().detect_speech(audio_chunk)
            
            # Convertir en format uniforme
            if isinstance(is_speech, bool):
                return {
                    'is_speech': is_speech,
                    'probability': 0.8 if is_speech else 0.2
                }
            elif isinstance(is_speech, dict):
                return is_speech
            else:
                return {'is_speech': False, 'probability': 0.0}
                
        except Exception as e:
            logging.warning(f"⚠️ Erreur détection base: {e}")
            return {'is_speech': False, 'probability': 0.0}

    async def _update_context(self, detection_result: Dict[str, Any], energy_level: float):
        """Mise à jour du contexte conversationnel"""
        with self.lock:
            # Ajouter à l'historique
            self.detection_history.append(detection_result.copy())
            self.energy_history.append(energy_level)
            
            # Mettre à jour le contexte
            if detection_result.get('is_speech', False):
                self.conversation_context.last_speech_time = time.time()
            
            # Baseline d'énergie adaptatif
            if len(self.energy_history) > 10:
                self.conversation_context.energy_baseline = np.mean(list(self.energy_history))

    async def _adapt_thresholds(self, detection_result: Dict[str, Any], energy_level: float):
        """Adaptation des seuils basée sur la performance"""
        # Adaptation simple basée sur l'historique récent
        if len(self.detection_history) < 20:
            return
        
        recent_detections = list(self.detection_history)[-20:]
        speech_rate = sum(1 for r in recent_detections if r.get('is_speech', False)) / len(recent_detections)
        
        # Ajuster le seuil de parole
        if speech_rate > 0.8:  # Trop de détections, augmenter seuil
            self.current_speech_threshold = min(0.9, self.current_speech_threshold + self.threshold_adaptation_rate)
            self.advanced_metrics["threshold_adaptations"] += 1
        elif speech_rate < 0.1:  # Pas assez de détections, diminuer seuil
            self.current_speech_threshold = max(0.1, self.current_speech_threshold - self.threshold_adaptation_rate)
            self.advanced_metrics["threshold_adaptations"] += 1

    async def _update_cache(self, audio_chunk: np.ndarray, result: Dict[str, Any]):
        """Mise à jour du cache intelligent"""
        try:
            chunk_hash = hash(audio_chunk.tobytes())
            
            # Éviter de surcharger le cache
            if len(self.segment_cache) > 1000:
                # Supprimer les entrées les moins utilisées
                sorted_entries = sorted(
                    self.segment_cache.items(),
                    key=lambda x: x[1]['usage_count']
                )
                for key, _ in sorted_entries[:100]:  # Supprimer 100 entrées
                    del self.segment_cache[key]
            
            # Ajouter nouvelle entrée
            self.segment_cache[chunk_hash] = {
                'result': result.copy(),
                'timestamp': time.time(),
                'usage_count': 1
            }
            
        except Exception as e:
            logging.warning(f"⚠️ Erreur mise à jour cache: {e}")

    def get_conversation_insights(self) -> Dict[str, Any]:
        """Retourne des insights sur la conversation en cours"""
        insights = {
            'total_chunks_processed': len(self.detection_history),
            'speech_percentage': 0.0,
            'avg_energy_level': 0.0,
            'conversation_duration_minutes': self.conversation_context.conversation_duration / 60,
            'cache_efficiency': 0.0,
            'threshold_adaptations': self.advanced_metrics["threshold_adaptations"],
            'current_thresholds': {
                'speech': self.current_speech_threshold,
                'energy': self.current_energy_threshold
            }
        }
        
        if len(self.detection_history) > 0:
            speech_count = sum(1 for r in self.detection_history if r.get('is_speech', False))
            insights['speech_percentage'] = (speech_count / len(self.detection_history)) * 100
        
        if len(self.energy_history) > 0:
            insights['avg_energy_level'] = float(np.mean(list(self.energy_history)))
        
        total_operations = len(self.detection_history)
        if total_operations > 0:
            insights['cache_efficiency'] = (self.advanced_metrics["cache_hits"] / total_operations) * 100
        
        return insights

    def reset_conversation_context(self):
        """Remet à zéro le contexte conversationnel"""
        with self.lock:
            self.conversation_context = ConversationContext(
                speaker_patterns={},
                silence_patterns=[],
                energy_baseline=0.0,
                last_speech_time=None,
                conversation_duration=0.0
            )
            
            self.detection_history.clear()
            self.energy_history.clear()
            self.segment_cache.clear()
            
            # Reset des seuils
            self.current_speech_threshold = 0.5
            self.current_energy_threshold = 0.001
            
            # Reset des métriques
            for key in self.advanced_metrics:
                self.advanced_metrics[key] = 0
            
            logging.info("🔄 Contexte conversationnel remis à zéro")

    async def cleanup(self):
        """Nettoyage des ressources"""
        await super().cleanup()
        
        # Nettoyer le thread pool
        self.executor.shutdown(wait=True)
        
        # Vider les caches
        self.segment_cache.clear()
        
        logging.info("✅ VAD Optimized Manager RTX 3090 nettoyé")

async def test_vad_optimized_manager():
    """Test du VAD Optimized Manager"""
    print("🧪 Test VAD Optimized Manager RTX 3090")
    
    # Test avec différentes configurations
    manager = VADOptimizedManager(
        chunk_ms=32,
        context_window_size=20,
        adaptive_thresholds=True,
        enable_caching=True
    )
    
    print("🚀 Initialisation...")
    await manager.initialize_optimized()
    
    # Test avec audio synthétique
    print("🎤 Test détection...")
    for i in range(10):
        # Alterner silence et parole simulée
        if i % 2 == 0:
            audio_chunk = np.random.randn(512).astype(np.float32) * 0.01  # Silence
        else:
            audio_chunk = np.random.randn(512).astype(np.float32) * 0.5   # Parole
        
        result = await manager.detect_speech_optimized(audio_chunk)
        print(f"   Chunk {i}: {result['is_speech']} (prob: {result['probability']:.2f}, source: {result['source']})")
    
    # Insights
    insights = manager.get_conversation_insights()
    print(f"\n📊 Insights: {insights}")
    
    # Nettoyage
    await manager.cleanup()
    print("✅ Test terminé")

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    asyncio.run(test_vad_optimized_manager()) 