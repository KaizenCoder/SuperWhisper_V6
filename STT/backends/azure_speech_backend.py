#!/usr/bin/env python3
"""
Backend Azure Speech Services pour SuperWhisper V6
🚀 Reconnaissance vocale streaming haute performance avec Azure Speech
"""

import os
import sys
import asyncio
import json
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, AsyncGenerator
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    raise ImportError("Azure Cognitive Services Speech SDK requis: pip install azure-cognitiveservices-speech")

# Configuration portable SuperWhisper V6
def _setup_portable_environment():
    """Configure l'environnement pour exécution portable"""
    import pathlib
    current_file = pathlib.Path(__file__).resolve()
    
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    else:
        project_root = current_file.parent.parent.parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    os.chdir(project_root)
    return project_root

_PROJECT_ROOT = _setup_portable_environment()

from STT.backends.base_stt_backend import BaseSTTBackend, STTResult

@dataclass
class AzureSpeechConfig:
    """Configuration Azure Speech Services"""
    speech_key: str
    speech_region: str
    language: str = "fr-FR"
    endpoint_id: Optional[str] = None  # Pour Custom Speech
    profanity_option: str = "Masked"
    enable_dictation: bool = True
    enable_detailed_results: bool = True
    enable_word_level_timestamps: bool = True
    continuous_recognition: bool = True
    segmentation_silence_timeout_ms: int = 500
    initial_silence_timeout_ms: int = 5000

class AzureSpeechBackend(BaseSTTBackend):
    """
    Backend Azure Speech Services optimisé pour SuperWhisper V6
    - Streaming temps réel
    - Reconnaissance continue
    - Qualité supérieure à Whisper
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration Azure
        self.azure_config = AzureSpeechConfig(
            speech_key=config.get('azure_speech_key', os.getenv('AZURE_SPEECH_KEY')),
            speech_region=config.get('azure_speech_region', os.getenv('AZURE_SPEECH_REGION', 'francecentral')),
            language=config.get('language', 'fr-FR'),
            endpoint_id=config.get('custom_endpoint_id'),
            continuous_recognition=config.get('continuous_recognition', True),
            segmentation_silence_timeout_ms=config.get('segmentation_silence_timeout_ms', 500),
            initial_silence_timeout_ms=config.get('initial_silence_timeout_ms', 5000)
        )
        
        if not self.azure_config.speech_key:
            raise ValueError("AZURE_SPEECH_KEY requis dans config ou variable d'environnement")
        if not self.azure_config.speech_region:
            raise ValueError("AZURE_SPEECH_REGION requis dans config ou variable d'environnement")
        
        # Configuration Speech SDK
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.azure_config.speech_key,
            region=self.azure_config.speech_region
        )
        
        self._setup_speech_config()
        
        # État du backend
        self.recognizer = None
        self.audio_stream = None
        self.is_recognizing = False
        self.recognition_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Callbacks streaming
        self.interim_callback: Optional[Callable[[str], None]] = None
        self.final_callback: Optional[Callable[[STTResult], None]] = None
        
        print(f"🎤 Azure Speech Backend initialisé: {self.azure_config.language} ({self.azure_config.speech_region})")
    
    def _setup_speech_config(self):
        """Configure le Speech SDK"""
        # Langue de reconnaissance
        self.speech_config.speech_recognition_language = self.azure_config.language
        
        # Format de sortie détaillé
        if self.azure_config.enable_detailed_results:
            self.speech_config.output_format = speechsdk.OutputFormat.Detailed
        
        # Profanity filtering
        if self.azure_config.profanity_option:
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceResponse_ProfanityOption,
                self.azure_config.profanity_option
            )
        
        # Custom Speech endpoint
        if self.azure_config.endpoint_id:
            self.speech_config.endpoint_id = self.azure_config.endpoint_id
        
        # Timeouts optimisés pour SuperWhisper V6
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(self.azure_config.segmentation_silence_timeout_ms)
        )
        
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            str(self.azure_config.initial_silence_timeout_ms)
        )
        
        # Enable word-level timestamps
        if self.azure_config.enable_word_level_timestamps:
            self.speech_config.request_word_level_timestamps()
        
        # Optimisations pour streaming
        self.speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_RecoMode, "INTERACTIVE")
    
    async def transcribe(self, audio: np.ndarray) -> STTResult:
        """
        Transcription audio simple (compatible interface SuperWhisper V6)
        """
        start_time = time.time()
        
        try:
            # Conversion audio vers format Azure
            audio_bytes = self._numpy_to_wav_bytes(audio)
            
            # Configuration audio
            audio_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            
            # Recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Push audio data
            audio_stream.write(audio_bytes)
            audio_stream.close()
            
            # Reconnaissance
            result = recognizer.recognize_once()
            processing_time = time.time() - start_time
            
            # Traitement du résultat
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Parse detailed results si disponible
                detailed_result = self._parse_detailed_result(result)
                
                return STTResult(
                    text=result.text,
                    confidence=detailed_result.get('confidence', 0.9),
                    segments=detailed_result.get('segments', []),
                    processing_time=processing_time,
                    device=self.device,
                    rtf=processing_time / (len(audio) / 16000),  # Real-Time Factor
                    backend_used="AzureSpeech",
                    success=True,
                    timestamp=time.time()
                )
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return STTResult(
                    text="",
                    confidence=0.0,
                    segments=[],
                    processing_time=processing_time,
                    device=self.device,
                    rtf=processing_time / (len(audio) / 16000),
                    backend_used="AzureSpeech",
                    success=False,
                    error="Aucune reconnaissance"
                )
            
            else:
                # Erreur de reconnaissance
                cancellation = speechsdk.CancellationDetails(result)
                error_msg = f"Erreur Azure Speech: {cancellation.reason}"
                if cancellation.error_details:
                    error_msg += f" - {cancellation.error_details}"
                
                return STTResult(
                    text="",
                    confidence=0.0,
                    segments=[],
                    processing_time=processing_time,
                    device=self.device,
                    rtf=processing_time / (len(audio) / 16000),
                    backend_used="AzureSpeech",
                    success=False,
                    error=error_msg
                )
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.total_errors += 1
            
            return STTResult(
                text="",
                confidence=0.0,
                segments=[],
                processing_time=processing_time,
                device=self.device,
                rtf=processing_time / (len(audio) / 16000) if len(audio) > 0 else 0.0,
                backend_used="AzureSpeech",
                success=False,
                error=f"Exception Azure Speech: {str(e)}"
            )
        
        finally:
            self.total_requests += 1
            self.total_processing_time += processing_time
    
    async def start_continuous_recognition(
        self,
        interim_callback: Optional[Callable[[str], None]] = None,
        final_callback: Optional[Callable[[STTResult], None]] = None
    ) -> bool:
        """
        Démarre la reconnaissance continue optimisée pour SuperWhisper V6
        """
        if self.is_recognizing:
            return False
        
        with self.recognition_lock:
            try:
                # Callbacks
                self.interim_callback = interim_callback
                self.final_callback = final_callback
                
                # Audio stream pour input continu
                self.audio_stream = speechsdk.audio.PushAudioInputStream()
                audio_config = speechsdk.audio.AudioConfig(stream=self.audio_stream)
                
                # Recognizer continu
                self.recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config,
                    audio_config=audio_config
                )
                
                # Event handlers
                self.recognizer.recognizing.connect(self._on_recognizing)
                self.recognizer.recognized.connect(self._on_recognized)
                self.recognizer.canceled.connect(self._on_canceled)
                self.recognizer.session_stopped.connect(self._on_session_stopped)
                
                # Start recognition
                self.recognizer.start_continuous_recognition()
                self.is_recognizing = True
                
                print(f"🎤 Reconnaissance continue Azure Speech démarrée")
                return True
                
            except Exception as e:
                print(f"❌ Erreur démarrage reconnaissance continue: {e}")
                return False
    
    async def stop_continuous_recognition(self):
        """Arrête la reconnaissance continue"""
        if not self.is_recognizing:
            return
        
        with self.recognition_lock:
            try:
                if self.recognizer:
                    self.recognizer.stop_continuous_recognition()
                    self.recognizer = None
                
                if self.audio_stream:
                    self.audio_stream.close()
                    self.audio_stream = None
                
                self.is_recognizing = False
                print(f"🎤 Reconnaissance continue Azure Speech arrêtée")
                
            except Exception as e:
                print(f"❌ Erreur arrêt reconnaissance: {e}")
    
    async def push_audio(self, audio: np.ndarray):
        """
        Push audio pour reconnaissance continue
        Compatible avec le streaming SuperWhisper V6
        """
        if not self.is_recognizing or not self.audio_stream:
            return False
        
        try:
            # Conversion audio
            audio_bytes = self._numpy_to_wav_bytes(audio, add_header=False)
            
            # Push vers Azure
            self.audio_stream.write(audio_bytes)
            return True
            
        except Exception as e:
            print(f"❌ Erreur push audio: {e}")
            return False
    
    def _on_recognizing(self, evt):
        """Handler pour résultats intermédiaires"""
        if self.interim_callback and evt.result.text:
            self.interim_callback(evt.result.text)
    
    def _on_recognized(self, evt):
        """Handler pour résultats finaux"""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and self.final_callback:
            # Parse résultat détaillé
            detailed_result = self._parse_detailed_result(evt.result)
            
            result = STTResult(
                text=evt.result.text,
                confidence=detailed_result.get('confidence', 0.9),
                segments=detailed_result.get('segments', []),
                processing_time=0.1,  # Estimation pour streaming
                device=self.device,
                rtf=0.1,  # Real-time pour streaming
                backend_used="AzureSpeech",
                success=True,
                timestamp=time.time()
            )
            
            self.final_callback(result)
    
    def _on_canceled(self, evt):
        """Handler pour annulations"""
        print(f"❌ Reconnaissance annulée: {evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"❌ Détails erreur: {evt.error_details}")
    
    def _on_session_stopped(self, evt):
        """Handler pour arrêt de session"""
        print(f"📴 Session Azure Speech arrêtée")
    
    def health_check(self) -> bool:
        """Vérifie l'état de santé du backend"""
        try:
            # Test simple de configuration
            test_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config)
            return True
        except Exception as e:
            print(f"❌ Health check échoué: {e}")
            return False
    
    def _numpy_to_wav_bytes(self, audio: np.ndarray, add_header: bool = True) -> bytes:
        """Convertit numpy array vers bytes WAV"""
        # Normalisation et conversion 16-bit PCM
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        if add_header:
            # Header WAV simple pour Azure
            import struct
            sample_rate = 16000
            channels = 1
            bits_per_sample = 16
            
            # WAV header
            header = struct.pack('<CCCCIHHHIHHHCCCC',
                b'R', b'I', b'F', b'F',
                36 + len(audio) * 2,  # File size
                b'W', b'A', b'V', b'E',
                b'f', b'm', b't', b' ',
                16,  # Subchunk1Size
                1,   # AudioFormat (PCM)
                channels,
                sample_rate,
                sample_rate * channels * bits_per_sample // 8,  # ByteRate
                channels * bits_per_sample // 8,  # BlockAlign
                bits_per_sample,
                b'd', b'a', b't', b'a',
                len(audio) * 2  # Subchunk2Size
            )
            
            return header + audio.tobytes()
        else:
            return audio.tobytes()
    
    def _parse_detailed_result(self, result) -> Dict[str, Any]:
        """Parse les résultats détaillés d'Azure Speech"""
        try:
            # Tenter de parser le JSON détaillé
            if hasattr(result, 'json') and result.json:
                detailed = json.loads(result.json)
                
                # Extraire segments avec timestamps
                segments = []
                if 'NBest' in detailed and detailed['NBest']:
                    best_result = detailed['NBest'][0]
                    confidence = best_result.get('Confidence', 0.9)
                    
                    if 'Words' in best_result:
                        for word_info in best_result['Words']:
                            segments.append({
                                'text': word_info.get('Word', ''),
                                'start': word_info.get('Offset', 0) / 10_000_000,  # Convert to seconds
                                'end': (word_info.get('Offset', 0) + word_info.get('Duration', 0)) / 10_000_000,
                                'confidence': word_info.get('Confidence', confidence)
                            })
                
                return {
                    'confidence': confidence,
                    'segments': segments
                }
        
        except Exception as e:
            print(f"⚠️ Erreur parsing résultat détaillé: {e}")
        
        return {
            'confidence': 0.9,
            'segments': [{'text': result.text, 'start': 0, 'end': 0, 'confidence': 0.9}]
        }
    
    def __del__(self):
        """Cleanup du backend"""
        if self.is_recognizing:
            try:
                asyncio.create_task(self.stop_continuous_recognition())
            except:
                pass
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Factory function pour SuperWhisper V6
def create_azure_speech_backend(config: Dict[str, Any]) -> AzureSpeechBackend:
    """Factory pour créer un backend Azure Speech"""
    return AzureSpeechBackend(config) 