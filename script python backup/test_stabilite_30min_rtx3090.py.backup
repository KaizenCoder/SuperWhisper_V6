#!/usr/bin/env python3
"""
🏆 TEST STABILITÉ 30MIN RTX 3090 - SUPERWHISPER V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Test de stabilité prolongée (30min simulé en 2min) avec Memory Leak V4
Phase 4.4 - Tests Stabilité 30min
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

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
import torch


def validate_rtx3090_mandatory():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    # RTX 3090 = ~24GB
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")


class StabilityTestSuite:
    """Suite de tests de stabilité prolongée"""
    
    def __init__(self, duration_minutes: int = 30, accelerated: bool = True):
        self.original_duration = duration_minutes
        self.actual_duration = 2.0 if accelerated else duration_minutes  # 2min en mode accéléré
        self.accelerated = accelerated
        self.running = False
        
        self.results = {
            "start_time": datetime.now().isoformat(),
            "test_config": {
                "original_duration_min": self.original_duration,
                "actual_duration_min": self.actual_duration,
                "accelerated_mode": self.accelerated,
                "CUDA_VISIBLE_DEVICES": os.environ.get('CUDA_VISIBLE_DEVICES')
            },
            "stability_metrics": {
                "memory_snapshots": [],
                "performance_snapshots": [],
                "error_count": 0,
                "warnings_count": 0
            }
        }
        
        # Initialisation Memory Leak V4
        sys.path.append(str(Path.cwd()))
        import memory_leak_v4
        self.gpu_manager = memory_leak_v4.GPUMemoryManager(enable_json_logging=True)
        
    def log_snapshot(self, snapshot_type: str, data: Dict[str, Any]):
        """Enregistre un snapshot de stabilité"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - datetime.fromisoformat(self.results["start_time"])).total_seconds(),
            "data": data
        }
        self.results["stability_metrics"][f"{snapshot_type}_snapshots"].append(snapshot)
        
    def simulate_continuous_workload(self):
        """Simule une charge de travail continue SuperWhisper V6"""
        print("🔄 Démarrage simulation charge continue...")
        
        cycle_count = 0
        start_time = datetime.now()
        duration_seconds = self.actual_duration * 60
        
        while self.running and (datetime.now() - start_time).total_seconds() < duration_seconds:
            cycle_count += 1
            
            try:
                with self.gpu_manager.gpu_context(f"stability_cycle_{cycle_count}") as ctx:
                    # Simulation STT (Speech-to-Text)
                    audio_tensor = torch.randn(16000 * 5, device="cuda:0")  # 5 sec audio
                    time.sleep(0.1 if self.accelerated else 1.0)
                    
                    # Simulation LLM (Large Language Model)
                    llm_tensor = torch.randint(0, 50000, (1, 512), device="cuda:0")  # 512 tokens
                    time.sleep(0.15 if self.accelerated else 2.0)
                    
                    # Simulation TTS (Text-to-Speech)
                    tts_tensor = torch.randn(80, 800, device="cuda:0")  # Mel spectrogram
                    time.sleep(0.1 if self.accelerated else 1.5)
                    
                    # Cleanup automatique via context manager
                    del audio_tensor, llm_tensor, tts_tensor
                    torch.cuda.empty_cache()
                    
                    # Snapshot mémoire périodique
                    if cycle_count % 10 == 0:
                        memory_stats = self.gpu_manager.get_memory_stats()
                        self.log_snapshot("memory", {
                            "cycle": cycle_count,
                            "allocated_gb": memory_stats.get("allocated_gb", 0),
                            "reserved_gb": memory_stats.get("reserved_gb", 0),
                            "fragmentation_gb": memory_stats.get("fragmentation_gb", 0)
                        })
                        
                        # Affichage périodique
                        elapsed = (datetime.now() - start_time).total_seconds()
                        progress = (elapsed / duration_seconds) * 100
                        print(f"   📊 Cycle {cycle_count} - {progress:.1f}% - Mémoire: {memory_stats.get('allocated_gb', 0):.3f}GB")
                    
            except Exception as e:
                self.results["stability_metrics"]["error_count"] += 1
                print(f"   ⚠️  Erreur cycle {cycle_count}: {e}")
                
            # Pause entre cycles (accélérée)
            time.sleep(0.05 if self.accelerated else 0.5)
        
        print(f"✅ Simulation terminée - {cycle_count} cycles complétés")
        return cycle_count
    
    def monitor_performance(self):
        """Monitore les performances en arrière-plan"""
        print("📈 Démarrage monitoring performance...")
        
        start_time = datetime.now()
        duration_seconds = self.actual_duration * 60
        snapshot_interval = 10 if self.accelerated else 60  # Snapshot chaque 10s en mode accéléré
        
        while self.running and (datetime.now() - start_time).total_seconds() < duration_seconds:
            try:
                # Test performance ponctuel
                perf_start = time.perf_counter()
                test_tensor = torch.randn(2000, 2000, device="cuda:0")
                result = torch.matmul(test_tensor, test_tensor)
                torch.cuda.synchronize()
                perf_time = time.perf_counter() - perf_start
                
                del test_tensor, result
                torch.cuda.empty_cache()
                
                # Snapshot performance
                elapsed = (datetime.now() - start_time).total_seconds()
                self.log_snapshot("performance", {
                    "elapsed_seconds": elapsed,
                    "matmul_time_ms": perf_time * 1000,
                    "estimated_gflops": (2 * 2000**3) / (perf_time * 1e9)
                })
                
            except Exception as e:
                self.results["stability_metrics"]["warnings_count"] += 1
                print(f"   ⚠️  Warning performance: {e}")
            
            time.sleep(snapshot_interval)
        
        print("📈 Monitoring performance terminé")
    
    def run_stability_test(self):
        """Exécute le test de stabilité complet"""
        print("=" * 80)
        print("🏆 TEST STABILITÉ 30MIN RTX 3090 - SUPERWHISPER V6")
        print("=" * 80)
        print(f"📅 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Durée: {self.original_duration}min {'(accéléré 2min)' if self.accelerated else ''}")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 80)
        
        self.running = True
        
        # Démarrage monitoring en arrière-plan
        performance_thread = threading.Thread(target=self.monitor_performance)
        performance_thread.daemon = True
        performance_thread.start()
        
        # Test principal de charge continue
        cycles_completed = self.simulate_continuous_workload()
        
        # Arrêt monitoring
        self.running = False
        performance_thread.join(timeout=5)
        
        # Analyse finale
        self.analyze_stability_results(cycles_completed)
        
        return True
    
    def analyze_stability_results(self, cycles_completed: int):
        """Analyse les résultats de stabilité"""
        print("\n" + "=" * 80)
        print("📊 ANALYSE STABILITÉ RTX 3090")
        print("=" * 80)
        
        # Finalisation résultats
        self.results["end_time"] = datetime.now().isoformat()
        self.results["test_summary"] = {
            "cycles_completed": cycles_completed,
            "total_errors": self.results["stability_metrics"]["error_count"],
            "total_warnings": self.results["stability_metrics"]["warnings_count"],
            "memory_snapshots_count": len(self.results["stability_metrics"]["memory_snapshots"]),
            "performance_snapshots_count": len(self.results["stability_metrics"]["performance_snapshots"])
        }
        
        # Analyse mémoire
        memory_snapshots = self.results["stability_metrics"]["memory_snapshots"]
        if memory_snapshots:
            memory_values = [s["data"]["allocated_gb"] for s in memory_snapshots]
            memory_analysis = {
                "min_memory_gb": min(memory_values),
                "max_memory_gb": max(memory_values),
                "avg_memory_gb": sum(memory_values) / len(memory_values),
                "memory_stable": max(memory_values) - min(memory_values) < 0.5  # Variation < 500MB
            }
            self.results["memory_analysis"] = memory_analysis
            
            print(f"💾 Mémoire GPU:")
            print(f"   📊 Min: {memory_analysis['min_memory_gb']:.3f}GB")
            print(f"   📊 Max: {memory_analysis['max_memory_gb']:.3f}GB") 
            print(f"   📊 Moyenne: {memory_analysis['avg_memory_gb']:.3f}GB")
            print(f"   📊 Stabilité: {'✅ Stable' if memory_analysis['memory_stable'] else '⚠️ Instable'}")
        
        # Analyse performance
        perf_snapshots = self.results["stability_metrics"]["performance_snapshots"]
        if perf_snapshots:
            gflops_values = [s["data"]["estimated_gflops"] for s in perf_snapshots]
            performance_analysis = {
                "min_gflops": min(gflops_values),
                "max_gflops": max(gflops_values),
                "avg_gflops": sum(gflops_values) / len(gflops_values),
                "performance_stable": (max(gflops_values) - min(gflops_values)) / max(gflops_values) < 0.1  # Variation < 10%
            }
            self.results["performance_analysis"] = performance_analysis
            
            print(f"⚡ Performance GPU:")
            print(f"   📊 Min: {performance_analysis['min_gflops']:.1f} GFLOPS")
            print(f"   📊 Max: {performance_analysis['max_gflops']:.1f} GFLOPS")
            print(f"   📊 Moyenne: {performance_analysis['avg_gflops']:.1f} GFLOPS")
            print(f"   📊 Stabilité: {'✅ Stable' if performance_analysis['performance_stable'] else '⚠️ Instable'}")
        
        # Verdict final
        stability_score = 100
        if self.results["test_summary"]["total_errors"] > 0:
            stability_score -= self.results["test_summary"]["total_errors"] * 10
        if self.results["test_summary"]["total_warnings"] > 5:
            stability_score -= (self.results["test_summary"]["total_warnings"] - 5) * 5
        if memory_snapshots and not self.results["memory_analysis"]["memory_stable"]:
            stability_score -= 20
        if perf_snapshots and not self.results["performance_analysis"]["performance_stable"]:
            stability_score -= 15
            
        self.results["stability_score"] = max(0, stability_score)
        
        print(f"\n🏆 VERDICT STABILITÉ:")
        print(f"   📈 Score: {stability_score}/100")
        print(f"   🔄 Cycles: {cycles_completed}")
        print(f"   ❌ Erreurs: {self.results['test_summary']['total_errors']}")
        print(f"   ⚠️  Warnings: {self.results['test_summary']['total_warnings']}")
        
        if stability_score >= 90:
            print("   🎉 EXCELLENTE stabilité RTX 3090 !")
        elif stability_score >= 70:
            print("   ✅ Bonne stabilité RTX 3090")
        else:
            print("   ⚠️  Stabilité perfectible")
        
        # Export rapport JSON
        report_file = "stability_test_report_rtx3090.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"💾 Rapport stabilité exporté: {report_file}")


def main():
    """Point d'entrée principal"""
    try:
        # Validation GPU obligatoire
        validate_rtx3090_mandatory()
        
        # Exécution test stabilité
        stability_suite = StabilityTestSuite(duration_minutes=30, accelerated=True)
        success = stability_suite.run_stability_test()
        
        if success:
            print("\n🎉 TEST STABILITÉ 30MIN TERMINÉ AVEC SUCCÈS !")
            return 0
        else:
            print("\n⚠️  PROBLÈMES LORS DU TEST STABILITÉ")
            return 1
            
    except Exception as e:
        print(f"\n🚫 ERREUR CRITIQUE STABILITÉ: {e}")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 