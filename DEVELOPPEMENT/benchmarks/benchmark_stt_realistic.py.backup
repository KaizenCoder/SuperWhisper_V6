#!/usr/bin/env python3
"""
Benchmark STT Réaliste - Luxa v1.1
===================================
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Teste les performances STT avec insanely-fast-whisper et faster-whisper
avec mapping GPU RTX 3090 exclusif et configuration réaliste.
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

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Maintenant imports normaux...
import time
import numpy as np
import torch
import asyncio
from typing import Dict, Any

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

try:
    from insanely_fast_whisper.transcribe import Transcriber
except ImportError:
    Transcriber = None
    print("⚠️ insanely-fast-whisper non installé")

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
    print("⚠️ faster-whisper non installé")

class STTBenchmark:
    def __init__(self):
        # Validation RTX 3090 obligatoire
        validate_rtx3090_mandatory()
        
        # RTX 3090 est maintenant CUDA:0 (seule visible)
        self.device_index = 0  # RTX 3090 mapping CUDA:0
        print(f"🎯 Utilisation RTX 3090 (CUDA:0) pour STT")
        
    async def benchmark_insanely_fast_whisper(self):
        """Test réel avec insanely-fast-whisper"""
        print(f"\n🎯 Testing insanely-fast-whisper on RTX 3090 (CUDA:0)")
        
        if Transcriber is None:
            print("❌ insanely-fast-whisper non disponible")
            return float('inf')
        
        try:
            # Configuration réaliste sur RTX 3090
            transcriber = Transcriber(
                model_name="openai/whisper-large-v3",
                device_id=self.device_index,  # RTX 3090 = CUDA:0
                torch_dtype="float16",  # Pas INT8 direct
                batch_size=4,
                better_tokenization=True
            )
            
            # Audio test 3 secondes
            test_audio = np.random.randn(48000).astype(np.float32)
            
            # Warmup
            print("🔥 Warmup...")
            for _ in range(3):
                _ = transcriber.transcribe(test_audio)
            
            # Mesure latence
            print("📊 Mesure des performances...")
            latencies = []
            for i in range(10):
                start = time.time()
                segments = transcriber.transcribe(test_audio)
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                print(f"   Run {i+1}: {latency:.1f}ms")
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            print(f"✅ insanely-fast-whisper:")
            print(f"   Latence moyenne: {avg_latency:.1f} ± {std_latency:.1f}ms")
            
            # Vérifier VRAM RTX 3090
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(self.device_index) / 1024**3
                print(f"   VRAM RTX 3090: {vram_used:.2f}GB")
            
            return avg_latency
            
        except Exception as e:
            print(f"❌ Erreur insanely-fast-whisper: {e}")
            return float('inf')
            
    async def benchmark_faster_whisper(self):
        """Alternative avec faster-whisper (quantification INT8)"""
        print(f"\n🎯 Testing faster-whisper INT8 on RTX 3090 (CUDA:0)")
        
        if WhisperModel is None:
            print("❌ faster-whisper non disponible")
            return float('inf')
        
        try:
            # Modèle avec quantification INT8 réelle sur RTX 3090
            model = WhisperModel(
                "large-v3",
                device="cuda",    # RTX 3090 automatiquement (seul device visible)
                device_index=self.device_index,  # RTX 3090 = CUDA:0
                compute_type="int8_float16",  # Quantification INT8 supportée
                num_workers=1,
                download_root="./models"
            )
            
            # Audio test
            test_audio = np.random.randn(48000).astype(np.float32)
            
            # Warmup
            print("🔥 Warmup...")
            for _ in range(3):
                segments, _ = model.transcribe(test_audio, beam_size=1)
                _ = list(segments)
            
            # Benchmark avec chunks streaming
            print("📊 Benchmark streaming...")
            latencies = []
            chunk_size = 16000  # 1 seconde
            
            for i in range(0, len(test_audio), chunk_size):
                chunk = test_audio[i:i+chunk_size]
                start = time.time()
                segments, _ = model.transcribe(chunk, beam_size=1)
                # Consommer le générateur
                _ = list(segments)
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            print(f"✅ faster-whisper INT8:")
            print(f"   Latence moyenne: {avg_latency:.1f} ± {std_latency:.1f}ms")
            
            # Vérifier VRAM RTX 3090
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(self.device_index) / 1024**3
                print(f"   VRAM RTX 3090: {vram_used:.2f}GB")
            
            return avg_latency
            
        except Exception as e:
            print(f"❌ Erreur faster-whisper: {e}")
            return float('inf')
    
    async def run_full_benchmark(self):
        """Lance tous les benchmarks STT"""
        print("🚀 LUXA v1.1 - Benchmark STT Réaliste - RTX 3090")
        print("="*50)
        
        # Vérifier CUDA
        if not torch.cuda.is_available():
            print("❌ CUDA non disponible")
            return {}
        
        print(f"🔧 Configuration GPU RTX 3090:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"   GPU {i}: {props.name} ({free/1024**3:.1f}/{total/1024**3:.1f}GB libre)")
        
        results = {}
        
        # Test insanely-fast-whisper
        results["insanely_fast"] = await self.benchmark_insanely_fast_whisper()
        
        # Test faster-whisper
        results["faster_whisper"] = await self.benchmark_faster_whisper()
        
        # Résumé
        print("\n📊 RÉSULTATS FINAUX:")
        print("="*30)
        for method, latency in results.items():
            if latency != float('inf'):
                status = "🟢" if latency < 500 else "🟡" if latency < 1000 else "🔴"
                print(f"{status} {method}: {latency:.1f}ms")
            else:
                print(f"🔴 {method}: ÉCHEC")
        
        # Recommandation
        best_method = min(results.items(), key=lambda x: x[1])
        if best_method[1] != float('inf'):
            print(f"\n🏆 Recommandé: {best_method[0]} ({best_method[1]:.1f}ms)")
        
        return results

async def main():
    """Point d'entrée principal"""
    benchmark = STTBenchmark()
    results = await benchmark.run_full_benchmark()
    
    # Sauvegarder résultats
    import json
    with open("benchmark_stt_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Résultats sauvés dans benchmark_stt_results.json")

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    asyncio.run(main()) 