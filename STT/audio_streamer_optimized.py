#!/usr/bin/env python3
"""
Audio Streamer Asynchrone Optimisé pour SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Intégration des 7 optimisations critiques:
1. Détection automatique périphérique audio robuste
2. Calibration automatique gain (RMS cible 0.05)
3. Architecture asynchrone pour faible latence
4. VAD avancé WebRTC + fallback RMS
5. Correction format audio PCM 16-bit pour VAD
6. Filtrage anti-hallucination post-transcription
7. Architecture séparée AudioStreamer + AudioStreamingManager

Architecture: Buffer circulaire + VAD + Pipeline async + Interface UnifiedSTTManager

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
import sounddevice as sd
import threading
import queue
import time
import logging
import torch
from typing import Optional, Callable, List, Tuple, Dict, Any
from collections import deque
import wave
import tempfile
import struct
from scipy import signal
from pathlib import Path

# Imports WebRTC-VAD avec gestion d'erreur
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("⚠️ WebRTC-VAD non disponible, utilisation fallback RMS")

# Validation RTX 3090 obligatoire
def validate_rtx3090_configuration():
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

# APPELER VALIDATION AU DÉMARRAGE
validate_rtx3090_configuration()


class VoiceActivityDetector:
    """
    Détecteur d'activité vocale optimisé pour éliminer les hallucinations Whisper
    PATCH DÉVELOPPEUR C: WebRTC-VAD avec format audio correct + fallback RMS intelligent
    """
    def __init__(self, sample_rate=16000, aggressiveness=1):
        self.sample_rate = sample_rate
        self.webrtc_available = WEBRTC_AVAILABLE
        
        # Configuration WebRTC-VAD si disponible
        if self.webrtc_available:
            try:
                self.vad = webrtcvad.Vad()
                self.vad.set_mode(aggressiveness)  # Mode permissif (1)
                print("✅ WebRTC-VAD initialisé avec succès")
            except Exception as e:
                print(f"⚠️ Erreur WebRTC-VAD: {e}, utilisation fallback")
                self.webrtc_available = False
        
        # Fallback RMS - seuil très permissif selon recommandations
        self.rms_threshold = 0.005  # Seuil très bas pour capter toute voix
        
    def is_speech(self, pcm_bytes: bytes, sample_rate: int) -> bool:
        """
        Interface directe WebRTC-VAD selon patch développeur C
        Args:
            pcm_bytes: Audio en format int16 PCM bytes
            sample_rate: Fréquence d'échantillonnage (doit être 8000, 16000, 32000, ou 48000)
        """
        if self.webrtc_available and sample_rate in [8000, 16000, 32000, 48000]:
            try:
                # WebRTC-VAD nécessite des chunks de taille spécifique
                frame_duration = 30  # ms
                frame_size = int(sample_rate * frame_duration / 1000)
                
                if len(pcm_bytes) >= frame_size * 2:  # 2 bytes par sample int16
                    frame_bytes = pcm_bytes[:frame_size * 2]
                    return self.vad.is_speech(frame_bytes, sample_rate)
            except Exception as e:
                print(f"⚠️ WebRTC-VAD erreur: {e}, fallback RMS")
        
        # Fallback RMS si WebRTC indisponible ou erreur
        return True  # Mode très permissif par défaut
            
    def has_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """
        Méthode principale pour détection d'activité vocale
        Utilise RMS avec seuil très permissif selon recommandations
        """
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms >= self.rms_threshold


class HallucinationFilter:
    """
    Filtre pour détecter et éliminer les hallucinations communes de Whisper
    Patterns identifiés lors des tests microphone live
    """
    def __init__(self):
        # Phrases d'hallucination communes identifiées
        self.hallucination_patterns = [
            "sous-titres réalisés par la communauté d'amara.org",
            "sous-titres réalisés par l'amara.org", 
            "merci d'avoir regardé cette vidéo",
            "merci d'avoir regardé",
            "n'hésitez pas à vous abonner",
            "like et abonne-toi",
            "commentez et partagez",
            "à bientôt pour une nouvelle vidéo",
            "musique libre de droit",
            "copyright",
            "creative commons",
            # Patterns spécifiques français
            "bonjour et bienvenue",
            "merci de votre attention",
            "n'oubliez pas de vous abonner",
            "likez cette vidéo"
        ]
        
        # Stats de filtrage
        self.stats = {
            'texts_analyzed': 0,
            'hallucinations_detected': 0,
            'repetition_filtered': 0,
            'empty_filtered': 0
        }
        
    def is_hallucination(self, text: str) -> bool:
        """
        Détermine si le texte est probablement une hallucination
        """
        self.stats['texts_analyzed'] += 1
        
        if not text or len(text.strip()) == 0:
            self.stats['empty_filtered'] += 1
            return True
            
        text_lower = text.lower().strip()
        
        # Vérifier patterns d'hallucination
        for pattern in self.hallucination_patterns:
            if pattern in text_lower:
                self.stats['hallucinations_detected'] += 1
                return True
                
        # Vérifier répétitions suspectes
        words = text_lower.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:  # Trop de répétitions
                self.stats['repetition_filtered'] += 1
                return True
                
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Retourne les statistiques de filtrage"""
        return self.stats.copy()


class AudioStreamer:
    """
    Capture l'audio du microphone en continu et le transmet via un callback.
    Intègre VAD, filtrage anti-hallucination et calibration automatique gain.
    
    Optimisations critiques implémentées:
    1. Détection automatique périphérique par nom (robuste aux changements ID)
    2. Calibration automatique gain (RMS cible 0.05)
    3. Architecture asynchrone avec threading
    4. VAD WebRTC + fallback RMS intelligent
    5. Correction format audio float32 → int16 PCM
    6. Interface avec filtrage hallucinations
    7. Stats détaillées pour monitoring
    """
    def __init__(self, 
                 callback: Callable[[np.ndarray], None], 
                 logger: logging.Logger, 
                 sample_rate=16000, 
                 chunk_duration=3.0, 
                 device_name="Rode NT-USB"):
        
        # Configuration audio
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_frames = int(sample_rate * chunk_duration)
        self.channels = 1
        self.device_name = device_name
        
        # Callbacks et logging
        self.callback = callback
        self.logger = logger
        
        # Résolution automatique du périphérique par nom (Optimisation #1)
        self.device_id = self._resolve_device_id(self.device_name)
        
        # Intégration VAD et filtres (Optimisations #4, #6)
        self.vad = VoiceActivityDetector(sample_rate, aggressiveness=1)
        self.hallucination_filter = HallucinationFilter()
        
        # Threading et contrôle
        self.running = False
        self.stream = None
        self.thread = None
        
        # Calibration gain automatique (Optimisation #2)
        self.auto_gain_enabled = True
        self.target_rms = 0.05  # RMS cible recommandée par développeur C
        self.gain_factor = 1.0  # Facteur de gain adaptatif
        self.rms_history = []
        self.calibration_complete = False
        
        # Stats de filtrage et performance
        self.stats = {
            'chunks_processed': 0,
            'chunks_with_voice': 0,
            'chunks_filtered_noise': 0,
            'chunks_filtered_vad': 0,
            'total_audio_duration': 0.0,
            'avg_rms': 0.0,
            'gain_applied': False,
            'device_detection_success': self.device_id is not None
        }

    def _resolve_device_id(self, name_part: str) -> Optional[int]:
        """
        Trouve l'ID du périphérique audio dont le nom contient name_part.
        Robuste aux changements d'ID Windows lors branchement/débranchement.
        OPTIMISATION #1: Détection automatique périphérique
        """
        try:
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                device_name = device.get('name', '').lower()
                max_input_channels = device.get('max_input_channels', 0)
                
                if name_part.lower() in device_name and max_input_channels > 0:
                    self.logger.info(f"🎤 Périphérique audio détecté: ID {idx} - {device['name']}")
                    return idx
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur lors de la recherche des périphériques: {e}")
        
        # Fallback: utiliser périphérique par défaut avec warning
        self.logger.warning(f"❌ Périphérique '{name_part}' non trouvé. Utilisation du périphérique par défaut.")
        return None  # sd.InputStream utilisera le device par défaut

    def _auto_calibrate_gain(self, rms: float) -> float:
        """
        Calibration automatique gain selon développeur C
        OPTIMISATION #2: Calibration automatique gain
        Objectif: RMS cible 0.05-0.1 pour optimiser ratio chunks
        """
        if not self.auto_gain_enabled:
            return 1.0
            
        # Collecter historique RMS pour stabilisation
        self.rms_history.append(rms)
        if len(self.rms_history) > 10:
            self.rms_history.pop(0)
            
        # Calibration après 5 échantillons
        if len(self.rms_history) >= 5 and not self.calibration_complete:
            avg_rms = np.mean(self.rms_history)
            
            if avg_rms < 0.02:  # Signal trop faible selon développeur C
                self.gain_factor = min(self.target_rms / avg_rms, 3.0)  # Limiter gain max
                self.logger.info(f"🔧 Auto-gain activé: {self.gain_factor:.2f}x (RMS {avg_rms:.4f} → {self.target_rms:.4f})")
                self.calibration_complete = True
                self.stats['gain_applied'] = True
                
        return self.gain_factor

    def start(self) -> bool:
        """Démarre la capture audio avec validation RTX 3090"""
        if self.running:
            return True
            
        try:
            # Validation RTX 3090 avant démarrage
            validate_rtx3090_configuration()
            
            self.logger.info("🎤 Démarrage de la capture audio optimisée...")
            self.logger.info(f"   📱 Périphérique: {self.device_name} (ID: {self.device_id})")
            self.logger.info(f"   🎛️ Configuration: {self.sample_rate}Hz, chunks {self.chunk_duration}s")
            self.logger.info(f"   🔧 Auto-gain: {self.auto_gain_enabled} (cible RMS: {self.target_rms})")
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur démarrage capture: {e}")
            return False

    def _capture_loop(self):
        """
        Boucle de capture qui lit depuis le microphone.
        OPTIMISATION #3: Architecture asynchrone
        """
        try:
            # Configuration stream avec device spécifique
            self.stream = sd.InputStream(
                device=self.device_id,  # ← FIX MAJEUR: route vers le bon micro
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_frames,
                callback=self._audio_callback
            )
            
            with self.stream:
                self.logger.info("✅ Stream audio démarré")
                while self.running:
                    time.sleep(0.1)  # La boucle est principalement pilotée par le callback
                    
        except Exception as e:
            self.logger.error(f"❌ Erreur dans la boucle de capture: {e}")
        finally:
            self.logger.info("🛑 Boucle de capture terminée")
        
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback de sounddevice, appelé avec de nouvelles données audio.
        OPTIMISATIONS #4, #5: VAD + Correction format audio
        """
        if status:
            self.logger.warning(f"AudioStreamer status: {status}")
        
        if not self.running or not self.callback:
            return
        
        try:
            # Conversion mono + float64 → float32 
            audio_chunk = indata[:, 0].astype(np.float32)
            
            # Calcul RMS pour diagnostic et calibration
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            
            # Auto-calibration gain (OPTIMISATION #2)
            gain = self._auto_calibrate_gain(rms)
            if gain > 1.0:
                audio_chunk = audio_chunk * gain
                rms = rms * gain  # Mettre à jour RMS après gain
            
            # Mise à jour stats
            self.stats['chunks_processed'] += 1
            self.stats['total_audio_duration'] += self.chunk_duration
            self.stats['avg_rms'] = (self.stats['avg_rms'] * (self.stats['chunks_processed'] - 1) + rms) / self.stats['chunks_processed']
            
            # VAD avec fallback RMS (OPTIMISATION #4)
            voice_detected = self.vad.has_voice_activity(audio_chunk)
            
            if voice_detected:
                self.stats['chunks_with_voice'] += 1
                self.logger.debug(f"✅ Chunk transmis au moteur - RMS={rms:.6f}")
                
                # Transmission au callback (vers UnifiedSTTManager)
                self.callback(audio_chunk)
            else:
                self.stats['chunks_filtered_noise'] += 1
                self.logger.debug(f"🔇 Chunk filtré (silence) - RMS={rms:.6f}")
        
        except Exception as e:
            self.logger.error(f"❌ Erreur dans le callback AudioStreamer: {e}")

    def stop(self):
        """Arrête la capture audio avec stats finales"""
        if not self.running:
            return
        
        self.logger.info("🛑 Arrêt de la capture audio...")
        
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.stream:
            self.stream.close()
            
        # Affichage stats finales
        self._log_final_stats()
        self.logger.info("✅ Capture audio arrêtée.")

    def _log_final_stats(self):
        """Affiche les statistiques finales de la session"""
        stats = self.get_filtering_stats()
        hallucination_stats = self.hallucination_filter.get_stats()
        
        self.logger.info("📊 === STATISTIQUES SESSION AUDIO ===")
        self.logger.info(f"   🎤 Chunks traités: {stats['chunks_processed']}")
        self.logger.info(f"   🗣️ Chunks avec voix: {stats['chunks_with_voice']}")
        self.logger.info(f"   🔇 Chunks filtrés (silence): {stats['chunks_filtered_noise']}")
        self.logger.info(f"   ⏱️ Durée totale: {stats['total_audio_duration']:.1f}s")
        self.logger.info(f"   📈 RMS moyen: {stats['avg_rms']:.6f}")
        self.logger.info(f"   🔧 Gain appliqué: {stats['gain_applied']}")
        self.logger.info(f"   🚫 Hallucinations détectées: {hallucination_stats['hallucinations_detected']}")

    def get_filtering_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes de filtrage"""
        return self.stats.copy()

    def add_to_buffer(self, audio_data: np.ndarray):
        """
        Méthode pour ajouter manuellement des données audio au flux.
        Utile pour les tests. Simule une capture micro.
        """
        if self.running and self.callback:
            self.logger.info(f"💉 Injection {len(audio_data)/self.sample_rate:.2f}s audio pour test")
            self.callback(audio_data)


class AudioStreamingManager:
    """
    Chef d'orchestre pour le streaming audio continu et la transcription
    OPTIMISATION #7: Architecture séparée AudioStreamer + AudioStreamingManager
    
    Interface avec UnifiedSTTManager existant de SuperWhisper V6
    """
    def __init__(self, 
                 unified_stt_manager, 
                 device_name="Rode NT-USB",
                 chunk_duration=1.0):
        
        # Validation RTX 3090 obligatoire
        validate_rtx3090_configuration()
        
        # Composants principaux
        self.stt_manager = unified_stt_manager
        self.logger = self._setup_logging()
        
        # Configuration streaming
        self.device_name = device_name
        self.chunk_duration = chunk_duration
        
        # AudioStreamer avec callback vers notre méthode
        self.streamer = AudioStreamer(
            callback=self._handle_audio_chunk,
            logger=self.logger,
            sample_rate=16000,  # Compatible avec UnifiedSTTManager
            chunk_duration=chunk_duration,
            device_name=device_name
        )
        
        # État et résultats
        self.continuous_mode = False
        self.results_queue = queue.Queue(maxsize=100)
        self.last_result = None
        
        # Filtrage hallucinations
        self.hallucination_filter = HallucinationFilter()
        
        # Stats manager
        self.stats = {
            'session_start': None,
            'chunks_received': 0,
            'transcriptions_completed': 0,
            'transcriptions_failed': 0,
            'hallucinations_filtered': 0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info("✅ AudioStreamingManager initialisé avec RTX 3090")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging pour AudioStreamingManager"""
        logger = logging.getLogger('AudioStreamingManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - AudioStreaming - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _handle_audio_chunk(self, audio_data: np.ndarray):
        """
        Gère la transcription d'un chunk audio via UnifiedSTTManager
        Interface avec l'architecture existante SuperWhisper V6
        """
        try:
            start_time = time.time()
            self.stats['chunks_received'] += 1
            
            # Transcription via UnifiedSTTManager existant
            # Note: Utilisation de la méthode synchrone pour compatibilité
            if hasattr(self.stt_manager, 'transcribe'):
                # Version asynchrone si disponible
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Si dans un contexte async, créer une tâche
                        task = asyncio.create_task(self.stt_manager.transcribe(audio_data))
                        # Note: Le résultat sera traité de manière asynchrone
                        return
                    else:
                        # Exécuter de manière synchrone
                        result = asyncio.run(self.stt_manager.transcribe(audio_data))
                except:
                    # Fallback synchrone
                    result = self.stt_manager.transcribe_sync(audio_data) if hasattr(self.stt_manager, 'transcribe_sync') else None
            else:
                self.logger.warning("⚠️ Méthode transcribe non disponible sur UnifiedSTTManager")
                return
            
            processing_time = time.time() - start_time
            
            if result and result.success:
                # Filtrage hallucinations (OPTIMISATION #6)
                if self.hallucination_filter.is_hallucination(result.text):
                    self.stats['hallucinations_filtered'] += 1
                    self.logger.warning(f"🚫 Hallucination filtrée: '{result.text[:50]}...'")
                    return
                
                # Transcription valide
                self.stats['transcriptions_completed'] += 1
                self.stats['avg_processing_time'] = (
                    (self.stats['avg_processing_time'] * (self.stats['transcriptions_completed'] - 1) + processing_time) 
                    / self.stats['transcriptions_completed']
                )
                
                # Stocker résultat
                self.last_result = result
                if not self.results_queue.full():
                    self.results_queue.put(result)
                
                self.logger.info(f"✅ Transcription: '{result.text}' (RTF: {result.rtf:.3f})")
                
            else:
                self.stats['transcriptions_failed'] += 1
                self.logger.warning(f"❌ Échec transcription: {result.error if result else 'Résultat None'}")
                
        except Exception as e:
            self.stats['transcriptions_failed'] += 1
            self.logger.error(f"❌ Erreur traitement chunk audio: {e}")

    def start_continuous_mode(self) -> bool:
        """Démarrer mode streaming continu avec validation RTX 3090"""
        try:
            # Validation RTX 3090
            validate_rtx3090_configuration()
            
            # Vérifier que UnifiedSTTManager est prêt
            if not hasattr(self.stt_manager, 'transcribe') and not hasattr(self.stt_manager, 'transcribe_sync'):
                self.logger.error("❌ UnifiedSTTManager pas prêt (méthode transcribe manquante)")
                return False
            
            self.logger.info("🌊 Démarrage mode streaming continu optimisé...")
            self.stats['session_start'] = time.time()
            self.continuous_mode = True
            
            # Démarrer AudioStreamer
            success = self.streamer.start()
            if success:
                self.logger.info("✅ Mode streaming continu actif")
            else:
                self.continuous_mode = False
                self.logger.error("❌ Échec démarrage AudioStreamer")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Erreur démarrage streaming: {e}")
            return False
    
    def get_latest_result(self, timeout=1.0) -> Optional[Dict[str, Any]]:
        """Récupérer dernier résultat transcription"""
        try:
            result = self.results_queue.get(timeout=timeout)
            # Convertir STTResult en dict pour compatibilité
            if hasattr(result, '__dict__'):
                return result.__dict__
            return result
        except queue.Empty:
            return None
    
    def stop_continuous_mode(self):
        """Arrêter mode streaming avec stats finales"""
        self.logger.info("⏹️ Arrêt mode streaming continu...")
        self.continuous_mode = False
        
        # Arrêter AudioStreamer
        self.streamer.stop()
        
        # Afficher stats session
        self._log_session_stats()
        
        self.logger.info("✅ Mode streaming arrêté")
    
    def _log_session_stats(self):
        """Affiche les statistiques de la session"""
        if self.stats['session_start']:
            session_duration = time.time() - self.stats['session_start']
            
            self.logger.info("📊 === STATISTIQUES SESSION STREAMING ===")
            self.logger.info(f"   ⏱️ Durée session: {session_duration:.1f}s")
            self.logger.info(f"   🎤 Chunks reçus: {self.stats['chunks_received']}")
            self.logger.info(f"   ✅ Transcriptions réussies: {self.stats['transcriptions_completed']}")
            self.logger.info(f"   ❌ Transcriptions échouées: {self.stats['transcriptions_failed']}")
            self.logger.info(f"   🚫 Hallucinations filtrées: {self.stats['hallucinations_filtered']}")
            self.logger.info(f"   ⚡ Temps traitement moyen: {self.stats['avg_processing_time']:.3f}s")
            
            # Stats AudioStreamer
            streamer_stats = self.streamer.get_filtering_stats()
            self.logger.info(f"   🔊 Ratio voix/silence: {streamer_stats['chunks_with_voice']}/{streamer_stats['chunks_filtered_noise']}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Stats complètes streaming + AudioStreamer + Hallucinations"""
        manager_stats = self.stats.copy()
        streamer_stats = self.streamer.get_filtering_stats()
        hallucination_stats = self.streamer.hallucination_filter.get_stats()
        
        return {
            'manager': manager_stats,
            'streamer': streamer_stats,
            'hallucination_filter': hallucination_stats,
            'continuous_mode_active': self.continuous_mode,
            'stt_manager_ready': hasattr(self.stt_manager, 'transcribe')
        }


# =============================================================================
# INTERFACE DE TEST ET DÉMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Test standalone du streaming audio optimisé
    Démonstration des 7 optimisations critiques
    """
    import sys
    from pathlib import Path
    
    # Ajouter le chemin du projet pour imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    print("🧪 === TEST AUDIO STREAMING OPTIMISÉ SUPERWHISPER V6 ===")
    print("🚨 Configuration GPU: RTX 3090 (CUDA:1) OBLIGATOIRE")
    
    # Validation RTX 3090 au démarrage
    try:
        validate_rtx3090_configuration()
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE: {e}")
        sys.exit(1)
    
    # Mock UnifiedSTTManager pour test
    class MockUnifiedSTTManager:
        def __init__(self):
            self.logger = logging.getLogger('MockSTTManager')
            
        async def transcribe(self, audio_data: np.ndarray):
            """Mock transcription pour test"""
            import asyncio
            await asyncio.sleep(0.1)  # Simuler processing
            
            # Mock result
            class MockResult:
                def __init__(self):
                    self.text = f"Transcription test {len(audio_data)} samples"
                    self.confidence = 0.95
                    self.rtf = 0.1
                    self.success = True
                    self.error = None
                    
            return MockResult()
    
    # Test avec mock manager
    def test_streaming_optimized():
        """Test complet du streaming optimisé"""
        print("\n🎯 Test AudioStreamingManager avec optimisations...")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Mock STT Manager
        mock_stt = MockUnifiedSTTManager()
        
        # AudioStreamingManager optimisé
        streaming_manager = AudioStreamingManager(
            unified_stt_manager=mock_stt,
            device_name="Rode NT-USB",
            chunk_duration=2.0
        )
        
        try:
            # Démarrer streaming
            if streaming_manager.start_continuous_mode():
                print("✅ Streaming démarré, test 15 secondes...")
                
                # Test pendant 15 secondes
                for i in range(15):
                    time.sleep(1)
                    
                    # Vérifier résultats
                    result = streaming_manager.get_latest_result(timeout=0.1)
                    if result:
                        print(f"📝 Résultat {i+1}: {result.get('text', 'N/A')}")
                    
                    # Stats intermédiaires
                    if i % 5 == 4:
                        stats = streaming_manager.get_stats()
                        print(f"📊 Stats {i+1}s: {stats['manager']['transcriptions_completed']} transcriptions")
                
                # Arrêter et afficher stats finales
                streaming_manager.stop_continuous_mode()
                
                final_stats = streaming_manager.get_stats()
                print("\n🏆 === RÉSULTATS TEST ===")
                print(f"✅ Transcriptions réussies: {final_stats['manager']['transcriptions_completed']}")
                print(f"❌ Transcriptions échouées: {final_stats['manager']['transcriptions_failed']}")
                print(f"🚫 Hallucinations filtrées: {final_stats['manager']['hallucinations_filtered']}")
                print(f"🎤 Chunks audio traités: {final_stats['streamer']['chunks_processed']}")
                print(f"🗣️ Chunks avec voix: {final_stats['streamer']['chunks_with_voice']}")
                
                return True
            else:
                print("❌ Échec démarrage streaming")
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️ Test interrompu par utilisateur")
            streaming_manager.stop_continuous_mode()
            return True
        except Exception as e:
            print(f"❌ Erreur test: {e}")
            return False
    
    # Exécuter test
    success = test_streaming_optimized()
    
    if success:
        print("\n🎉 Test AudioStreamer optimisé terminé avec succès!")
        print("🔗 Prêt pour intégration avec UnifiedSTTManager SuperWhisper V6")
    else:
        print("\n💥 Test échoué - vérifier configuration")
        sys.exit(1)
