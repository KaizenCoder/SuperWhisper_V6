#!/usr/bin/env python3
"""
Test Microphone Live - Solution STT Optimisée
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
🎤 TEST CRITIQUE: Conditions réelles microphone live
⚠️ ATTENTION: Ce test révèle les VRAIES performances (vs fichier pré-enregistré)
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajout du chemin STT au PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import asyncio
import time
import json
import numpy as np
import torch
import sounddevice as sd
import wave
from pathlib import Path
from datetime import datetime
import logging
import threading
import queue

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")

def list_audio_devices():
    """Liste les périphériques audio disponibles"""
    print(f"\n🎤 PÉRIPHÉRIQUES AUDIO DISPONIBLES")
    print("-" * 60)
    
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            print(f"   [{i}] {device['name']}")
            print(f"       Canaux: {device['max_input_channels']}, SR: {device['default_samplerate']:.0f}Hz")
    
    return input_devices

def select_microphone():
    """Sélection interactive du microphone"""
    devices = list_audio_devices()
    
    if not devices:
        raise RuntimeError("❌ Aucun périphérique d'entrée audio trouvé")
    
    print(f"\n🎯 SÉLECTION MICROPHONE")
    print("-" * 40)
    
    while True:
        try:
            choice = input(f"Choisissez le microphone [0-{len(devices)-1}] ou ENTER pour défaut: ").strip()
            
            if choice == "":
                device_id = None
                device_name = "Défaut système"
                break
            else:
                device_id = int(choice)
                if 0 <= device_id < len(devices):
                    device_name = devices[device_id][1]['name']
                    break
                else:
                    print(f"❌ Choix invalide. Entrez un nombre entre 0 et {len(devices)-1}")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
    
    print(f"✅ Microphone sélectionné: {device_name}")
    return device_id, device_name

class MicrophoneRecorder:
    """Enregistreur microphone avec buffer circulaire"""
    
    def __init__(self, device_id=None, sample_rate=16000, channels=1):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recorded_data = []
        
    def audio_callback(self, indata, frames, time, status):
        """Callback pour l'enregistrement audio"""
        if status:
            print(f"⚠️ Status audio: {status}")
        
        if self.is_recording:
            # Conversion en float32 et mono
            audio_data = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
            self.audio_queue.put(audio_data.copy())
            self.recorded_data.append(audio_data.copy())
    
    def start_recording(self):
        """Démarre l'enregistrement"""
        self.is_recording = True
        self.recorded_data = []
        
        self.stream = sd.InputStream(
            device=self.device_id,
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=self.audio_callback,
            blocksize=1024,
            dtype=np.float32
        )
        self.stream.start()
        print("🔴 Enregistrement démarré...")
    
    def stop_recording(self):
        """Arrête l'enregistrement"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Concaténation des données
        if self.recorded_data:
            full_audio = np.concatenate(self.recorded_data)
            print(f"⏹️ Enregistrement arrêté. Durée: {len(full_audio)/self.sample_rate:.2f}s")
            return full_audio
        else:
            print("⚠️ Aucune donnée audio enregistrée")
            return np.array([])
    
    def save_recording(self, audio_data, filename):
        """Sauvegarde l'enregistrement en WAV"""
        if len(audio_data) == 0:
            print("⚠️ Pas de données à sauvegarder")
            return False
        
        try:
            # Normalisation et conversion en int16
            audio_normalized = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            print(f"💾 Enregistrement sauvegardé: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return False

async def test_backend_avec_microphone(audio_data: np.ndarray, duration: float):
    """Test du backend optimisé avec audio microphone"""
    print(f"\n⚙️ TEST BACKEND OPTIMISÉ - MICROPHONE LIVE")
    print("-" * 50)
    
    try:
        from STT.backends.prism_stt_backend_optimized import OptimizedPrismSTTBackend
        
        config = {
            'model': 'large-v2',
            'compute_type': 'float16',
            'language': 'fr',  # Forçage français
            'beam_size': 10,   # Beam search optimisé
            'vad_filter': True,
            'vad_parameters': {
                'threshold': 0.2,
                'min_speech_duration_ms': 50,
                'max_speech_duration_s': float('inf'),
                'min_silence_duration_ms': 2000,
                'speech_pad_ms': 800
            }
        }
        
        print("🚀 Initialisation backend optimisé...")
        backend = OptimizedPrismSTTBackend(config)
        
        print("🎯 Transcription audio microphone...")
        start_time = time.perf_counter()
        
        result = await backend.transcribe(audio_data)
        
        processing_time = time.perf_counter() - start_time
        
        if result['success']:
            print(f"\n📝 RÉSULTATS TRANSCRIPTION MICROPHONE:")
            print(f"   Texte: \"{result['text']}\"")
            print(f"   Confiance: {result['confidence']:.3f}")
            print(f"   Segments: {len(result['segments'])}")
            print(f"   Temps traitement: {processing_time:.2f}s")
            print(f"   RTF: {result['rtf']:.3f}")
            print(f"   Mots transcrits: {len(result['text'].split())}")
            
            return result
        else:
            print(f"❌ Transcription échouée: {result.get('error', 'Erreur inconnue')}")
            return None
            
    except Exception as e:
        print(f"❌ Erreur test backend: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_post_processor_microphone(transcription: str):
    """Test du post-processeur avec transcription microphone"""
    print(f"\n🧪 TEST POST-PROCESSEUR - TRANSCRIPTION MICROPHONE")
    print("-" * 50)
    
    try:
        from STT.stt_postprocessor import STTPostProcessor
        
        processor = STTPostProcessor()
        
        print(f"📝 Texte original: \"{transcription}\"")
        
        start_time = time.time()
        processed, metrics = processor.process(transcription, 0.8)
        processing_time = (time.time() - start_time) * 1000  # ms
        
        print(f"📝 Texte corrigé: \"{processed}\"")
        print(f"   Corrections appliquées: {metrics['corrections_applied']}")
        print(f"   Boost confiance: +{metrics['confidence_boost']:.3f}")
        print(f"   Temps traitement: {processing_time:.1f}ms")
        
        return processed, metrics
        
    except Exception as e:
        print(f"❌ Erreur post-processeur: {e}")
        return transcription, {}

def calculate_wer(reference: str, hypothesis: str):
    """Calcul du Word Error Rate (WER)"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Matrice de distance d'édition
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + 1     # substitution
                )
    
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) * 100
    return wer, int(d[len(ref_words)][len(hyp_words)])

def generate_microphone_report(device_name: str, audio_duration: float, backend_result: dict, 
                             post_processed: str, metrics: dict, reference_text: str = None):
    """Génération du rapport de test microphone"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_microphone_live_optimise_report_{timestamp}.json"
    
    # Calcul WER si texte de référence fourni
    wer_data = {}
    if reference_text and backend_result:
        wer, errors = calculate_wer(reference_text, backend_result.get('text', ''))
        wer_data = {
            'reference_text': reference_text,
            'wer_percentage': wer,
            'edit_distance': errors,
            'reference_words': len(reference_text.split()),
            'transcribed_words': len(backend_result.get('text', '').split())
        }
    
    report = {
        'timestamp': timestamp,
        'test_type': 'microphone_live_reel',
        'warning': 'Résultats RÉELS - conditions microphone live',
        'microphone_info': {
            'device_name': device_name,
            'duration_seconds': audio_duration,
            'sample_rate': 16000,
            'channels': 1
        },
        'gpu_config': {
            'device': torch.cuda.get_device_name(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'cuda_version': torch.version.cuda
        },
        'backend_result': backend_result,
        'post_processing': {
            'original_text': backend_result.get('text', '') if backend_result else '',
            'processed_text': post_processed,
            'metrics': metrics
        },
        'performance': {
            'rtf': backend_result.get('rtf', 0) if backend_result else 0,
            'processing_time': backend_result.get('processing_time', 0) if backend_result else 0,
            'confidence': backend_result.get('confidence', 0) if backend_result else 0
        },
        'wer_analysis': wer_data
    }
    
    # Sauvegarde rapport
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Rapport microphone sauvegardé: {report_file}")
    return report_file

def display_test_instructions():
    """Affiche les instructions pour le test"""
    print(f"\n📋 INSTRUCTIONS TEST MICROPHONE LIVE")
    print("=" * 60)
    print("🎯 OBJECTIF: Tester la solution optimisée en conditions réelles")
    print("🎤 PROCÉDURE:")
    print("   1. Sélectionnez votre microphone")
    print("   2. Préparez le texte à lire (fourni ci-dessous)")
    print("   3. Appuyez sur ENTER pour démarrer l'enregistrement")
    print("   4. Lisez le texte clairement au microphone")
    print("   5. Appuyez sur ENTER pour arrêter")
    print("   6. Attendez la transcription et l'analyse")
    print()
    print("⚠️ IMPORTANT:")
    print("   → Parlez naturellement (pas trop lentement)")
    print("   → Gardez une distance normale du microphone")
    print("   → Environnement normal (pas de silence parfait)")
    print("   → Ce test révèle les VRAIES performances")
    print()

def get_test_text():
    """Retourne le texte de test à lire"""
    return """Bonjour, ceci est un test de validation pour SuperWhisper version 6. 
Je vais maintenant énoncer plusieurs phrases de complexité croissante pour évaluer la précision de transcription.
Premièrement, des mots simples : chat, chien, maison, voiture, ordinateur, téléphone.
Deuxièmement, les phrases courtes : il fait beau aujourd'hui, le café est délicieux, j'aime la musique classique.
Troisièmement, des phrases plus complexes : l'intelligence artificielle transforme notre manière de travailler et communiquer dans le monde moderne.
Quatrièmement, des termes techniques : algorithme, machine learning, GPU RTX 3090, faster-whisper, quantification INT8, latence de transcription.
Cinquièmement, des nombres et dates : vingt-trois, quarante-sept, mille neuf cent quatre-vingt-quinze, le quinze janvier deux mille vingt-quatre.
Sixièmement, les mots difficiles : chrysanthème, anticonstitutionnellement, prestidigitateur, kakémono, yaourt.
Septièmement, une phrase longue et complexe : L'optimisation des performances de transcription vocale nécessite une approche méthodique combinant la sélection appropriée des modèles, l'ajustement des paramètres de traitement et l'implémentation d'algorithmes de post-traitement pour améliorer la qualité du résultat final.
Fin du test de validation."""

async def main():
    """Test principal microphone live"""
    print("🎤 TEST MICROPHONE LIVE - SOLUTION STT OPTIMISÉE")
    print("🚨 GPU RTX 3090 OBLIGATOIRE")
    print("🎯 ATTENTION: Test en CONDITIONS RÉELLES - Vraies performances")
    print("=" * 80)
    
    try:
        # 1. Validation GPU
        validate_rtx3090_configuration()
        
        # 2. Instructions
        display_test_instructions()
        
        # 3. Texte de référence
        reference_text = get_test_text()
        print("📝 TEXTE À LIRE:")
        print("-" * 40)
        print(reference_text)
        print("-" * 40)
        print()
        
        # 4. Sélection microphone
        device_id, device_name = select_microphone()
        
        # 5. Préparation enregistreur
        recorder = MicrophoneRecorder(device_id=device_id)
        
        # 6. Enregistrement
        input("🎤 Appuyez sur ENTER pour démarrer l'enregistrement...")
        recorder.start_recording()
        
        input("🔴 ENREGISTREMENT EN COURS... Lisez le texte puis appuyez sur ENTER pour arrêter...")
        audio_data = recorder.stop_recording()
        
        if len(audio_data) == 0:
            print("❌ Aucune donnée audio enregistrée")
            return False
        
        duration = len(audio_data) / 16000
        print(f"✅ Audio enregistré: {duration:.2f}s")
        
        # 7. Sauvegarde enregistrement
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = f"test_microphone_live_{timestamp}.wav"
        recorder.save_recording(audio_data, audio_file)
        
        # 8. Test backend optimisé
        backend_result = await test_backend_avec_microphone(audio_data, duration)
        
        if not backend_result:
            print("❌ Test backend échoué")
            return False
        
        # 9. Test post-processeur
        post_processed, post_metrics = await test_post_processor_microphone(backend_result['text'])
        
        # 10. Calcul WER
        wer, errors = calculate_wer(reference_text, backend_result['text'])
        
        # 11. Résumé global
        print("\n" + "="*80)
        print("📊 RÉSUMÉ TEST MICROPHONE LIVE - CONDITIONS RÉELLES")
        print("="*80)
        
        print(f"🎤 Microphone: {device_name}")
        print(f"⏱️ Durée: {duration:.2f}s")
        print(f"✅ Backend optimisé: Transcription réussie")
        print(f"✅ Post-processeur: {post_metrics.get('corrections_applied', 0)} corrections")
        
        # Métriques clés
        print(f"\n📈 MÉTRIQUES CLÉS - CONDITIONS RÉELLES:")
        print(f"   RTF: {backend_result['rtf']:.3f}")
        print(f"   Confiance: {backend_result['confidence']:.3f}")
        print(f"   Temps traitement: {backend_result['processing_time']:.2f}s")
        print(f"   Mots transcrits: {len(backend_result['text'].split())}")
        print(f"   Corrections post-proc: {post_metrics.get('corrections_applied', 0)}")
        
        # WER Analysis
        print(f"\n🎯 ANALYSE PRÉCISION (WER):")
        print(f"   WER: {wer:.1f}%")
        print(f"   Erreurs: {errors}/{len(reference_text.split())} mots")
        print(f"   Mots référence: {len(reference_text.split())}")
        print(f"   Mots transcrits: {len(backend_result['text'].split())}")
        
        # Évaluation performance
        if wer < 15:
            print(f"   🎉 EXCELLENT: WER < 15%")
        elif wer < 25:
            print(f"   ✅ BON: WER < 25%")
        elif wer < 40:
            print(f"   ⚠️ MOYEN: WER < 40%")
        else:
            print(f"   ❌ INSUFFISANT: WER > 40%")
        
        # 12. Génération rapport
        report_file = generate_microphone_report(
            device_name, duration, backend_result, 
            post_processed, post_metrics, reference_text
        )
        
        # 13. Comparaison avec fichier pré-enregistré
        print(f"\n🔍 COMPARAISON FICHIER vs MICROPHONE:")
        print("   📁 Fichier pré-enregistré: RTF 0.115, Confiance 92.5%")
        print(f"   🎤 Microphone live: RTF {backend_result['rtf']:.3f}, Confiance {backend_result['confidence']*100:.1f}%")
        
        rtf_diff = ((backend_result['rtf'] - 0.115) / 0.115) * 100
        conf_diff = ((backend_result['confidence'] - 0.925) / 0.925) * 100
        
        print(f"   📊 Différence RTF: {rtf_diff:+.1f}%")
        print(f"   📊 Différence Confiance: {conf_diff:+.1f}%")
        
        print(f"\n🎯 CONCLUSION TEST MICROPHONE LIVE:")
        print("   → Test en CONDITIONS RÉELLES terminé")
        print("   → Performances mesurées avec microphone live")
        print("   → WER calculé avec texte de référence")
        print("   → Comparaison fichier vs microphone disponible")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur test microphone: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Vérification dépendances
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ Module sounddevice requis: pip install sounddevice")
        sys.exit(1)
    
    # Exécution test
    success = asyncio.run(main())
    
    if success:
        print(f"\n✅ Test microphone live terminé avec succès")
        print("🎯 Performances RÉELLES mesurées en conditions microphone")
    else:
        print(f"\n❌ Test microphone live échoué") 