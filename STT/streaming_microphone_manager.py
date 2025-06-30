#!/usr/bin/env python3
"""
🎙️ STREAMING MICROPHONE MANAGER - SUPERWHISPER V6
Real-time microphone → VAD → STT manager optimisé RTX 3090

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

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
import logging
import time
import struct
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, List

import numpy as np

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Imports après configuration GPU
try:
    import sounddevice as sd  # pip install sounddevice>=0.4.7
    import webrtcvad          # pip install webrtcvad>=2.0.10
    import torch
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    print("💡 Installation requise: pip install sounddevice webrtcvad torch")
    sys.exit(1)

# =============================================================================
# VALIDATION RTX 3090 OBLIGATOIRE
# =============================================================================
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

# =============================================================================
# CONSTANTES OPTIMISÉES SUPERWHISPER V6
# =============================================================================
SAMPLE_RATE: int = 16_000                 # Hz - synchronisé avec STT
FRAME_MS: int = 30                        # Chaque frame VAD = 30ms
FRAME_BYTES: int = int(SAMPLE_RATE * FRAME_MS / 1000 * 2)  # 2 bytes/sample
VAD_MODE: int = 2                         # 0 = très sensible, 3 = agressif
VAD_SILENCE_AFTER_MS: int = 400           # Fin de parole après ce silence
MAX_SEGMENT_MS: int = 10_000              # Limite pour éviter segments trop longs

# Constantes dérivées
FRAMES_PER_SEGMENT_LIMIT = MAX_SEGMENT_MS // FRAME_MS
SILENCE_FRAMES = VAD_SILENCE_AFTER_MS // FRAME_MS

# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================
@dataclass
class AudioFrame:
    pcm: bytes              # PCM 16-bit little-endian mono
    timestamp: float        # Timestamp capture (secondes)

@dataclass
class SpeechSegment:
    pcm: bytes
    start_ts: float
    end_ts: float
    duration_ms: float

# =============================================================================
# RING BUFFER LOCK-FREE
# =============================================================================
class RingBuffer:
    """Ring buffer lock-free pour absorber le jitter audio"""
    
    def __init__(self, capacity_frames: int):
        self.capacity = capacity_frames
        self.buffer: Deque[AudioFrame] = deque(maxlen=capacity_frames)

    def push(self, frame: AudioFrame):
        """Ajouter frame (auto-drop ancien si plein)"""
        self.buffer.append(frame)

    def get_chunk(self, n_frames: int) -> List[AudioFrame]:
        """Récupérer les n dernières frames"""
        if len(self.buffer) < n_frames:
            return []
        return list(self.buffer)[-n_frames:]

    def clear(self):
        """Vider le buffer"""
        self.buffer.clear()

# =============================================================================
# DÉTECTION AUTOMATIQUE MICROPHONE RODE NT-USB
# =============================================================================
def detect_rode_microphone():
    """Détection automatique du microphone RODE NT-USB"""
    try:
        devices = sd.query_devices()
        rode_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                device_name = device['name'].upper()
                if 'RODE' in device_name and 'NT-USB' in device_name:
                    rode_devices.append((i, device['name']))
        
        if rode_devices:
            print(f"🎤 RODE NT-USB détectés: {len(rode_devices)} instances")
            for device_id, name in rode_devices:
                print(f"   Device {device_id}: {name}")
            
            # Tester chaque instance pour trouver celle qui fonctionne
            for device_id, name in rode_devices:
                try:
                    # Test rapide de capture
                    test_data = sd.rec(int(0.1 * SAMPLE_RATE), 
                                     samplerate=SAMPLE_RATE, 
                                     channels=1, 
                                     device=device_id,
                                     dtype='float32')
                    sd.wait()
                    
                    if np.max(np.abs(test_data)) > 0.001:  # Signal détecté
                        print(f"✅ RODE NT-USB fonctionnel: Device {device_id}")
                        return device_id
                        
                except Exception as e:
                    print(f"⚠️ Device {device_id} non fonctionnel: {e}")
                    continue
            
            print("⚠️ Aucun RODE NT-USB fonctionnel - utilisation device par défaut")
            return None
        else:
            print("⚠️ Aucun RODE NT-USB détecté - utilisation device par défaut")
            return None
            
    except Exception as e:
        print(f"❌ Erreur détection microphone: {e}")
        return None

# =============================================================================
# STREAMING MICROPHONE MANAGER PRINCIPAL
# =============================================================================
class StreamingMicrophoneManager:
    """
    Gestionnaire streaming microphone temps réel pour SuperWhisper V6
    
    Architecture: Microphone → VAD WebRTC → Segments → UnifiedSTTManager
    Performance: <800ms premier mot, <1.6s phrase complète
    """
    
    def __init__(
        self,
        stt_manager,  # UnifiedSTTManager instance
        device: Optional[int] = None,
        on_transcription: Optional[Callable[[str, SpeechSegment], None]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        # Validation RTX 3090 obligatoire
        validate_rtx3090_configuration()
        
        self.stt_manager = stt_manager
        self.device = device or detect_rode_microphone()
        self.on_transcription = on_transcription or self._default_transcription_callback
        self.loop = loop or asyncio.get_event_loop()

        # Initialisation VAD et buffers
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.ring = RingBuffer(capacity_frames=FRAMES_PER_SEGMENT_LIMIT)
        self._audio_queue: asyncio.Queue[AudioFrame] = asyncio.Queue()
        self._capture_stream: Optional[sd.InputStream] = None
        
        # Configuration logging
        self._log = logging.getLogger("StreamingMic")
        if not self._log.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s [🎙️ Mic] %(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            self._log.addHandler(handler)
            self._log.setLevel(logging.INFO)
        
        # Statistiques
        self.stats = {
            'segments_processed': 0,
            'total_audio_duration': 0.0,
            'total_transcription_time': 0.0,
            'average_latency': 0.0
        }
        
        print(f"🎙️ StreamingMicrophoneManager initialisé")
        print(f"   Device: {self.device}")
        print(f"   VAD Mode: {VAD_MODE}")
        print(f"   Silence threshold: {VAD_SILENCE_AFTER_MS}ms")

    def _default_transcription_callback(self, text: str, segment: SpeechSegment):
        """Callback par défaut pour affichage transcription"""
        latency_ms = (time.time() - segment.end_ts) * 1000
        print(f"🗣️ [{segment.duration_ms:.0f}ms] {text} (latence: {latency_ms:.0f}ms)")

    # =========================================================================
    # API PUBLIQUE
    # =========================================================================
    async def run(self):
        """Démarrage streaming microphone (Ctrl-C pour arrêter)"""
        self._start_capture()
        worker = asyncio.create_task(self._vad_worker())
        
        self._log.info("🎙️ Streaming microphone démarré (Ctrl-C pour arrêter)...")
        self._log.info(f"🎮 GPU: RTX 3090 configurée pour STT")
        
        try:
            while True:
                await asyncio.sleep(0.2)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self._log.info("🛑 Arrêt streaming microphone...")
        finally:
            self._stop_capture()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            self._print_stats()
            self._log.info("✅ Streaming microphone arrêté")

    def _print_stats(self):
        """Affichage statistiques finales"""
        if self.stats['segments_processed'] > 0:
            avg_latency = self.stats['average_latency']
            total_duration = self.stats['total_audio_duration']
            total_transcription = self.stats['total_transcription_time']
            
            print(f"\n📊 STATISTIQUES STREAMING MICROPHONE")
            print(f"   Segments traités: {self.stats['segments_processed']}")
            print(f"   Audio total: {total_duration:.1f}s")
            print(f"   Temps transcription: {total_transcription:.1f}s")
            print(f"   Latence moyenne: {avg_latency:.0f}ms")
            print(f"   RTF moyen: {total_transcription/total_duration:.3f}")

    # =========================================================================
    # CAPTURE AUDIO (THREAD PORTAUDIO)
    # =========================================================================
    def _start_capture(self):
        """Démarrage capture audio avec callback PortAudio"""
        def audio_callback(indata, frames, time_info, status):
            if status:
                self._log.warning(f"PortAudio status: {status}")
            
            # Conversion float32 [-1,1] → int16 pour WebRTC VAD
            pcm_int16 = (indata[:, 0] * 32768).clip(-32768, 32767).astype(np.int16)
            pcm_bytes = pcm_int16.tobytes()
            
            if len(pcm_bytes) != FRAME_BYTES:
                return  # Frame invalide, ignorer
            
            frame = AudioFrame(pcm=pcm_bytes, timestamp=time.time())
            
            # Ajout au ring buffer et queue async (thread-safe)
            self.ring.push(frame)
            try:
                self.loop.call_soon_threadsafe(self._audio_queue.put_nowait, frame)
            except RuntimeError:
                pass  # Loop fermée

        self._capture_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * FRAME_MS / 1000),  # frames par callback
            channels=1,
            dtype="float32",
            callback=audio_callback,
            device=self.device,
        )
        self._capture_stream.start()
        self._log.info(f"🎤 Capture audio démarrée (device: {self.device})")

    def _stop_capture(self):
        """Arrêt capture audio"""
        if self._capture_stream is not None:
            self._capture_stream.stop()
            self._capture_stream.close()
            self._capture_stream = None

    # =========================================================================
    # WORKER VAD ASYNCHRONE
    # =========================================================================
    async def _vad_worker(self):
        """Worker asynchrone: consommation frames + VAD + assemblage segments"""
        speech_frames: List[AudioFrame] = []
        num_silence = 0

        while True:
            frame = await self._audio_queue.get()
            is_speech = self.vad.is_speech(frame.pcm, SAMPLE_RATE)

            if is_speech:
                speech_frames.append(frame)
                num_silence = 0
                
                # Limite sécurité: segment trop long
                if len(speech_frames) >= FRAMES_PER_SEGMENT_LIMIT:
                    await self._flush_segment(speech_frames)
                    speech_frames = []
            else:
                if speech_frames:
                    num_silence += 1
                    if num_silence >= SILENCE_FRAMES:
                        # Fin du segment actuel
                        await self._flush_segment(speech_frames)
                        speech_frames = []
                        num_silence = 0

    async def _flush_segment(self, frames: List[AudioFrame]):
        """Traitement segment de parole complet"""
        if not frames:
            return
        
        start_ts = frames[0].timestamp
        end_ts = frames[-1].timestamp + (FRAME_MS / 1000)
        duration_ms = (end_ts - start_ts) * 1000
        pcm_bytes = b"".join(f.pcm for f in frames)

        self._log.info(
            f"🗣️ Segment parole {duration_ms:.0f}ms → STT ({len(pcm_bytes)//2} échantillons)"
        )

        # Transcription via UnifiedSTTManager sur RTX 3090
        try:
            tic = time.time()
            
            # Conversion PCM bytes → numpy array pour UnifiedSTTManager
            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcription asynchrone
            result = await self.stt_manager.transcribe(audio_array)
            
            toc = time.time()
            transcription_time = toc - tic
            latency_ms = transcription_time * 1000
            
            # Extraction texte du résultat
            text = result.text if hasattr(result, 'text') else str(result)
            text = text.strip()
            
            if text:
                self._log.info(f"✅ STT {latency_ms:.0f}ms: '{text}'")
                
                # Mise à jour statistiques
                self.stats['segments_processed'] += 1
                self.stats['total_audio_duration'] += duration_ms / 1000
                self.stats['total_transcription_time'] += transcription_time
                self.stats['average_latency'] = (
                    self.stats['average_latency'] * (self.stats['segments_processed'] - 1) + latency_ms
                ) / self.stats['segments_processed']
                
                # Callback utilisateur
                segment = SpeechSegment(
                    pcm=pcm_bytes, 
                    start_ts=start_ts, 
                    end_ts=end_ts,
                    duration_ms=duration_ms
                )
                self.on_transcription(text, segment)
            else:
                self._log.warning(f"⚠️ STT vide après {latency_ms:.0f}ms")
                
        except Exception as exc:
            self._log.error(f"❌ Erreur STT: {exc}")

# =============================================================================
# FONCTION UTILITAIRE POUR TESTS
# =============================================================================
async def test_streaming_microphone(stt_manager, duration_seconds: int = 30):
    """Test rapide du streaming microphone"""
    print(f"🧪 Test streaming microphone {duration_seconds}s")
    
    def transcription_callback(text: str, segment: SpeechSegment):
        print(f"📝 TRANSCRIPTION: '{text}' ({segment.duration_ms:.0f}ms)")
    
    mic_manager = StreamingMicrophoneManager(
        stt_manager=stt_manager,
        on_transcription=transcription_callback
    )
    
    # Test limité dans le temps
    try:
        task = asyncio.create_task(mic_manager.run())
        await asyncio.wait_for(task, timeout=duration_seconds)
    except asyncio.TimeoutError:
        print(f"✅ Test terminé après {duration_seconds}s")
    except KeyboardInterrupt:
        print("🛑 Test interrompu par utilisateur")

if __name__ == "__main__":
    # Test standalone
    print("🎙️ StreamingMicrophoneManager - Test standalone")
    print("⚠️ Nécessite UnifiedSTTManager pour fonctionner") 