#!/usr/bin/env python3
"""
🧪 TEST STREAMING MICROPHONE MANAGER - SUPERWHISPER V6
Validation du streaming microphone temps réel avec VAD WebRTC

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    import torch
    from STT.unified_stt_manager import UnifiedSTTManager
    from STT.streaming_microphone_manager import StreamingMicrophoneManager, SpeechSegment
except ImportError as e:
    print(f"❌ Erreur import: {e}")
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
# CALLBACK TRANSCRIPTION AVEC MÉTRIQUES
# =============================================================================
class TranscriptionMetrics:
    """Collecteur de métriques pour validation"""
    
    def __init__(self):
        self.transcriptions = []
        self.total_segments = 0
        self.total_audio_duration = 0.0
        self.total_latency = 0.0
        self.start_time = time.time()
    
    def on_transcription(self, text: str, segment: SpeechSegment):
        """Callback appelé pour chaque transcription"""
        current_time = time.time()
        latency_ms = (current_time - segment.end_ts) * 1000
        
        # Enregistrement transcription
        transcription_data = {
            'text': text,
            'duration_ms': segment.duration_ms,
            'latency_ms': latency_ms,
            'timestamp': current_time,
            'words_count': len(text.split()) if text else 0
        }
        self.transcriptions.append(transcription_data)
        
        # Mise à jour métriques
        self.total_segments += 1
        self.total_audio_duration += segment.duration_ms / 1000
        self.total_latency += latency_ms
        
        # Affichage temps réel
        print(f"\n🗣️ SEGMENT {self.total_segments}")
        print(f"   📝 Texte: '{text}'")
        print(f"   ⏱️ Durée: {segment.duration_ms:.0f}ms")
        print(f"   🚀 Latence: {latency_ms:.0f}ms")
        print(f"   📊 Mots: {transcription_data['words_count']}")
        print(f"   ⏰ Temps: {current_time - self.start_time:.1f}s")
    
    def get_final_report(self):
        """Génération rapport final"""
        if not self.transcriptions:
            return "❌ Aucune transcription obtenue"
        
        # Calculs statistiques
        avg_latency = self.total_latency / self.total_segments
        total_words = sum(t['words_count'] for t in self.transcriptions)
        total_test_duration = time.time() - self.start_time
        rtf = self.total_audio_duration / total_test_duration if total_test_duration > 0 else 0
        
        # Texte complet
        full_text = " ".join(t['text'] for t in self.transcriptions if t['text'])
        
        report = f"""
📊 RAPPORT FINAL STREAMING MICROPHONE
{'='*50}
🎯 Segments traités: {self.total_segments}
📝 Mots transcrits: {total_words}
⏱️ Durée audio totale: {self.total_audio_duration:.1f}s
🚀 Latence moyenne: {avg_latency:.0f}ms
⏰ Durée test: {total_test_duration:.1f}s
🎮 RTF: {rtf:.3f}

📝 TRANSCRIPTION COMPLÈTE:
{'-'*30}
{full_text}

🔍 DÉTAIL SEGMENTS:
{'-'*30}"""
        
        for i, t in enumerate(self.transcriptions, 1):
            report += f"\n{i}. [{t['duration_ms']:.0f}ms, {t['latency_ms']:.0f}ms] {t['text']}"
        
        return report

# =============================================================================
# FONCTION TEST PRINCIPAL
# =============================================================================
async def test_streaming_microphone_complete():
    """Test complet du streaming microphone avec métriques"""
    
    print("🧪 TEST STREAMING MICROPHONE MANAGER - SUPERWHISPER V6")
    print("="*60)
    
    # Validation GPU
    validate_rtx3090_configuration()
    
    try:
        # Initialisation STT Manager
        print("\n🚀 Initialisation UnifiedSTTManager...")
        
        config = {
            'timeout_per_minute': 10.0,
            'max_retries': 3,
            'cache_enabled': True,
            'circuit_breaker_enabled': True,
            'fallback_chain': ['prism_primary'],
            'backends': [
                {
                    'name': 'prism_primary',
                    'type': 'prism',
                    'model': 'large-v2',
                    'compute_type': 'float16',
                    'language': 'fr',
                    'beam_size': 5,
                    'vad_filter': True
                }
            ]
        }
        
        stt_manager = UnifiedSTTManager(config=config)
        print("✅ UnifiedSTTManager initialisé")
        
        # Initialisation métriques
        metrics = TranscriptionMetrics()
        
        # Initialisation StreamingMicrophoneManager
        print("\n🎙️ Initialisation StreamingMicrophoneManager...")
        mic_manager = StreamingMicrophoneManager(
            stt_manager=stt_manager,
            on_transcription=metrics.on_transcription
        )
        print("✅ StreamingMicrophoneManager initialisé")
        
        # Instructions utilisateur
        print(f"\n🎯 TEST STREAMING TEMPS RÉEL")
        print(f"{'='*40}")
        print(f"📋 Instructions:")
        print(f"   1. Parlez clairement au microphone")
        print(f"   2. Faites des pauses entre les phrases")
        print(f"   3. Testez différents types de contenu:")
        print(f"      - Phrases courtes simples")
        print(f"      - Phrases longues complexes")
        print(f"      - Mots techniques (GPU, RTX 3090, etc.)")
        print(f"      - Nombres et dates")
        print(f"   4. Appuyez Ctrl+C pour arrêter")
        print(f"\n⏰ Démarrage dans 3 secondes...")
        
        await asyncio.sleep(3)
        
        # Démarrage test streaming
        print("\n🎙️ STREAMING MICROPHONE DÉMARRÉ")
        print("🛑 Appuyez Ctrl+C pour arrêter le test")
        print("-" * 50)
        
        await mic_manager.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrompu par utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Rapport final
        print("\n" + "="*60)
        print(metrics.get_final_report())
        print("="*60)

# =============================================================================
# TEST RAPIDE (30 SECONDES)
# =============================================================================
async def test_streaming_quick():
    """Test rapide de 30 secondes"""
    
    print("🧪 TEST RAPIDE STREAMING MICROPHONE (30s)")
    print("="*50)
    
    validate_rtx3090_configuration()
    
    try:
        # Configuration minimale
        config = {
            'timeout_per_minute': 10.0,
            'fallback_chain': ['prism_primary'],
            'backends': [
                {
                    'name': 'prism_primary',
                    'type': 'prism',
                    'model': 'large-v2',
                    'compute_type': 'float16',
                    'language': 'fr'
                }
            ]
        }
        
        stt_manager = UnifiedSTTManager(config=config)
        metrics = TranscriptionMetrics()
        
        mic_manager = StreamingMicrophoneManager(
            stt_manager=stt_manager,
            on_transcription=metrics.on_transcription
        )
        
        print("\n🎙️ Test 30 secondes - parlez maintenant!")
        
        # Test avec timeout
        task = asyncio.create_task(mic_manager.run())
        await asyncio.wait_for(task, timeout=30.0)
        
    except asyncio.TimeoutError:
        print("\n✅ Test 30s terminé")
    except KeyboardInterrupt:
        print("\n🛑 Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
    finally:
        print("\n" + metrics.get_final_report())

# =============================================================================
# MAIN
# =============================================================================
async def main():
    """Point d'entrée principal"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        await test_streaming_quick()
    else:
        await test_streaming_microphone_complete()

if __name__ == "__main__":
    print("🎙️ StreamingMicrophoneManager - Test SuperWhisper V6")
    print("Usage:")
    print("  python test_microphone_streaming.py          # Test complet")
    print("  python test_microphone_streaming.py --quick  # Test 30s")
    print()
    
    asyncio.run(main()) 