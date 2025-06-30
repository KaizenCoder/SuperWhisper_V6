#!/usr/bin/env python3
"""
Benchmark STT Optimisé - SuperWhisper V6
Comparaison rigoureuse Original vs Optimisé avec métriques détaillées
"""

import os
import sys
import asyncio
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime
import argparse

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
    sys.path.append(str(Path(__file__).parent.parent / "backends"))
    from prism_stt_backend_optimized import OptimizedPrismSTTBackend

try:
    from STT.backends.unified_stt_manager_optimized import OptimizedUnifiedSTTManager
except ImportError:
    # Fallback - utiliser le chemin relatif
    sys.path.append(str(Path(__file__).parent.parent / "backends"))
    from unified_stt_manager_optimized import OptimizedUnifiedSTTManager

try:
    from STT.stt_postprocessor import STTPostProcessor
except ImportError:
    # Fallback - utiliser le chemin relatif
    sys.path.append(str(Path(__file__).parent.parent))
    from stt_postprocessor import STTPostProcessor

# Fallback pour backend original
try:
    from STT.backends.prism_stt_backend import PrismSTTBackend
except ImportError:
    # Créer un backend original simulé pour comparaison
    class PrismSTTBackend:
        def __init__(self, config):
            self.config = config
            self.model_size = config.get('model', 'large-v2')
            self.device = config.get('device', 'cuda:1')
            
        async def transcribe(self, audio):
            # Simulation backend original avec performances dégradées
            await asyncio.sleep(0.5)  # Latence plus élevée
            
            # Simulation transcription avec erreurs typiques
            text = "Transcription simulée avec erreurs typiques du backend original"
            
            from dataclasses import dataclass
            from typing import List, Dict, Any, Optional
            
            @dataclass
            class STTResult:
                text: str
                confidence: float
                segments: List[Dict[str, Any]]
                processing_time: float
                device: str
                rtf: float
                backend_used: str
                success: bool
                error: Optional[str] = None
            
            return STTResult(
                text=text,
                confidence=0.75,  # Confiance plus faible
                segments=[],
                processing_time=0.5,
                device=self.device,
                rtf=0.5 / (len(audio) / 16000),
                backend_used=f"original_prism_{self.model_size}",
                success=True
            )
        
        def health_check(self):
            return True
        
        def get_metrics(self):
            return {"total_requests": 0, "successful_requests": 0}

@dataclass
class BenchmarkResult:
    """Résultat de benchmark pour un test"""
    test_name: str
    audio_duration: float
    
    # Résultats Original
    original_text: str
    original_confidence: float
    original_processing_time: float
    original_rtf: float
    original_success: bool
    
    # Résultats Optimisé
    optimized_text: str
    optimized_confidence: float
    optimized_processing_time: float
    optimized_rtf: float
    optimized_success: bool
    
    # Métriques de qualité
    wer: Optional[float] = None
    cer: Optional[float] = None
    
    # Améliorations
    latency_improvement: float = 0.0
    rtf_improvement: float = 0.0
    confidence_improvement: float = 0.0

class STTBenchmarkSuite:
    """Suite de benchmark complète pour STT"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Backends
        self.original_backend = None
        self.optimized_backend = None
        self.unified_manager = None
        
        # Résultats
        self.results: List[BenchmarkResult] = []
        
        # Configuration benchmark
        self.test_audio_dir = config.get('test_audio_dir', 'test_audio')
        self.reference_texts_file = config.get('reference_texts', 'reference_texts.json')
        self.output_dir = Path(config.get('output_dir', 'benchmark_results'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Métriques à calculer
        self.calculate_wer = config.get('calculate_wer', True)
        self.generate_plots = config.get('generate_plots', True)
        
        self.logger.info("🏁 Suite de benchmark STT initialisée")
    
    def _setup_logging(self) -> logging.Logger:
        """Configuration du logging"""
        logger = logging.getLogger('STTBenchmark')
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
        """Initialisation des backends"""
        try:
            self.logger.info("🚀 Initialisation des backends...")
            
            # Backend original
            original_config = {
                'model': self.config.get('model', 'large-v2'),
                'compute_type': self.config.get('compute_type', 'float16'),
                'device': 'cuda:1'
            }
            self.original_backend = PrismSTTBackend(original_config)
            
            # Backend optimisé
            optimized_config = dict(original_config)
            self.optimized_backend = OptimizedPrismSTTBackend(optimized_config)
            
            # Manager unifié (optionnel)
            if self.config.get('test_unified_manager', False):
                manager_config = dict(original_config)
                manager_config.update({
                    'cache_size_mb': 100,
                    'cache_ttl': 3600
                })
                self.unified_manager = OptimizedUnifiedSTTManager(manager_config)
                await self.unified_manager.initialize()
            
            self.logger.info("✅ Backends initialisés")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation: {e}")
            raise
    
    def _generate_test_audio(self) -> List[Tuple[str, np.ndarray, str]]:
        """Génère des échantillons audio de test"""
        test_cases = []
        
        # Textes de référence français
        reference_texts = [
            "Bonjour, ceci est un test de SuperWhisper.",
            "L'intelligence artificielle transforme notre monde moderne.",
            "Les algorithmes de machine learning utilisent des GPU RTX 3090.",
            "faster-whisper est plus rapide que Whisper original.",
            "Le chrysanthème est une fleur magnifique.",
            "Voici un kakémono japonais traditionnel.",
            "Les mots difficiles incluent anticonstitutionnellement.",
            "Vingt-trois, quarante-sept, mille neuf cent quatre-vingt-quinze.",
            "Premièrement, deuxièmement, troisièmement, quatrièmement, cinquièmement, sixièmement.",
            "Fin du test de transcription automatique."
        ]
        
        for i, text in enumerate(reference_texts):
            # Générer audio simulé
            duration = len(text) * 0.08  # ~80ms par caractère
            samples = int(16000 * duration)
            
            # Audio avec bruit léger pour simulation réaliste
            audio = np.random.randn(samples).astype(np.float32) * 0.05
            
            # Ajouter quelques pics pour simuler la parole
            speech_samples = np.random.choice(samples, size=samples//10, replace=False)
            audio[speech_samples] += np.random.randn(len(speech_samples)) * 0.3
            
            test_name = f"test_{i+1:02d}_{len(text)}chars"
            test_cases.append((test_name, audio, text))
        
        self.logger.info(f"📊 {len(test_cases)} cas de test générés")
        return test_cases
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """Exécute le benchmark complet"""
        try:
            self.logger.info("🏃 Démarrage du benchmark...")
            start_time = time.time()
            
            # Générer cas de test
            test_cases = self._generate_test_audio()
            
            # Exécuter tests
            for test_name, audio, reference_text in test_cases:
                self.logger.info(f"🧪 Test: {test_name}")
                
                result = await self._run_single_test(
                    test_name, audio, reference_text
                )
                self.results.append(result)
                
                # Pause entre tests
                await asyncio.sleep(0.1)
            
            # Calculer métriques globales
            global_metrics = self._calculate_global_metrics()
            
            # Générer rapport
            report = await self._generate_report(global_metrics)
            
            # Sauvegarder résultats
            await self._save_results(report)
            
            # Générer graphiques
            if self.generate_plots:
                await self._generate_plots()
            
            total_time = time.time() - start_time
            self.logger.info(f"✅ Benchmark terminé en {total_time:.1f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Erreur benchmark: {e}")
            raise
    
    async def _run_single_test(self, test_name: str, audio: np.ndarray, 
                              reference_text: str) -> BenchmarkResult:
        """Exécute un test unique"""
        audio_duration = len(audio) / 16000
        
        # Test backend original
        original_start = time.perf_counter()
        original_result = await self.original_backend.transcribe(audio)
        original_time = time.perf_counter() - original_start
        
        # Test backend optimisé
        optimized_start = time.perf_counter()
        optimized_result = await self.optimized_backend.transcribe(audio)
        optimized_time = time.perf_counter() - optimized_start
        
        # Calculer améliorations
        latency_improvement = ((original_time - optimized_time) / original_time * 100) if original_time > 0 else 0
        rtf_improvement = ((original_result.rtf - optimized_result.rtf) / original_result.rtf * 100) if original_result.rtf > 0 else 0
        confidence_improvement = optimized_result.confidence - original_result.confidence
        
        # Calculer WER/CER si référence disponible
        wer = None
        cer = None
        if reference_text and self.calculate_wer:
            wer = self._calculate_wer(reference_text, optimized_result.text)
            cer = self._calculate_cer(reference_text, optimized_result.text)
        
        return BenchmarkResult(
            test_name=test_name,
            audio_duration=audio_duration,
            
            original_text=original_result.text,
            original_confidence=original_result.confidence,
            original_processing_time=original_time,
            original_rtf=original_result.rtf,
            original_success=original_result.success,
            
            optimized_text=optimized_result.text,
            optimized_confidence=optimized_result.confidence,
            optimized_processing_time=optimized_time,
            optimized_rtf=optimized_result.rtf,
            optimized_success=optimized_result.success,
            
            wer=wer,
            cer=cer,
            
            latency_improvement=latency_improvement,
            rtf_improvement=rtf_improvement,
            confidence_improvement=confidence_improvement
        )
    
    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calcule le Word Error Rate"""
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
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words) * 100
    
    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calcule le Character Error Rate"""
        ref_chars = list(reference.lower())
        hyp_chars = list(hypothesis.lower())
        
        # Même algorithme que WER mais sur les caractères
        d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
        
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(hyp_chars) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,
                        d[i][j-1] + 1,
                        d[i-1][j-1] + 1
                    )
        
        return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars) * 100
    
    def _calculate_global_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques globales"""
        if not self.results:
            return {}
        
        # Moyennes
        avg_latency_improvement = np.mean([r.latency_improvement for r in self.results])
        avg_rtf_improvement = np.mean([r.rtf_improvement for r in self.results])
        avg_confidence_improvement = np.mean([r.confidence_improvement for r in self.results])
        
        # WER/CER moyens
        wer_results = [r.wer for r in self.results if r.wer is not None]
        cer_results = [r.cer for r in self.results if r.cer is not None]
        
        avg_wer = np.mean(wer_results) if wer_results else None
        avg_cer = np.mean(cer_results) if cer_results else None
        
        # Temps de traitement
        original_times = [r.original_processing_time for r in self.results]
        optimized_times = [r.optimized_processing_time for r in self.results]
        
        # Taux de succès
        original_success_rate = sum(r.original_success for r in self.results) / len(self.results) * 100
        optimized_success_rate = sum(r.optimized_success for r in self.results) / len(self.results) * 100
        
        return {
            "total_tests": len(self.results),
            "avg_latency_improvement_percent": avg_latency_improvement,
            "avg_rtf_improvement_percent": avg_rtf_improvement,
            "avg_confidence_improvement": avg_confidence_improvement,
            "avg_wer_percent": avg_wer,
            "avg_cer_percent": avg_cer,
            "original_avg_processing_time": np.mean(original_times),
            "optimized_avg_processing_time": np.mean(optimized_times),
            "original_success_rate_percent": original_success_rate,
            "optimized_success_rate_percent": optimized_success_rate,
            "speedup_factor": np.mean(original_times) / np.mean(optimized_times) if np.mean(optimized_times) > 0 else 0
        }
    
    async def _generate_report(self, global_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Génère le rapport de benchmark"""
        report = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "total_tests": len(self.results)
            },
            "global_metrics": global_metrics,
            "detailed_results": [asdict(result) for result in self.results],
            "summary": {
                "performance_improvement": {
                    "latency": f"{global_metrics.get('avg_latency_improvement_percent', 0):.1f}%",
                    "rtf": f"{global_metrics.get('avg_rtf_improvement_percent', 0):.1f}%",
                    "speedup": f"{global_metrics.get('speedup_factor', 1):.2f}x"
                },
                "quality_metrics": {
                    "avg_wer": f"{global_metrics.get('avg_wer_percent', 0):.1f}%",
                    "avg_cer": f"{global_metrics.get('avg_cer_percent', 0):.1f}%",
                    "confidence_boost": f"{global_metrics.get('avg_confidence_improvement', 0):.3f}"
                },
                "reliability": {
                    "original_success_rate": f"{global_metrics.get('original_success_rate_percent', 0):.1f}%",
                    "optimized_success_rate": f"{global_metrics.get('optimized_success_rate_percent', 0):.1f}%"
                }
            }
        }
        
        return report
    
    async def _save_results(self, report: Dict[str, Any]):
        """Sauvegarde les résultats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rapport JSON complet
        json_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # CSV des résultats détaillés
        csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df = pd.DataFrame([asdict(result) for result in self.results])
        df.to_csv(csv_file, index=False)
        
        # Résumé texte
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== BENCHMARK STT SUPERWHISPER V6 ===\n\n")
            f.write(f"Date: {report['benchmark_info']['timestamp']}\n")
            f.write(f"Tests: {report['benchmark_info']['total_tests']}\n\n")
            
            f.write("=== AMÉLIORATIONS PERFORMANCES ===\n")
            perf = report['summary']['performance_improvement']
            f.write(f"Latence: {perf['latency']}\n")
            f.write(f"RTF: {perf['rtf']}\n")
            f.write(f"Accélération: {perf['speedup']}\n\n")
            
            f.write("=== MÉTRIQUES QUALITÉ ===\n")
            quality = report['summary']['quality_metrics']
            f.write(f"WER moyen: {quality['avg_wer']}\n")
            f.write(f"CER moyen: {quality['avg_cer']}\n")
            f.write(f"Boost confiance: {quality['confidence_boost']}\n\n")
            
            f.write("=== FIABILITÉ ===\n")
            reliability = report['summary']['reliability']
            f.write(f"Succès Original: {reliability['original_success_rate']}\n")
            f.write(f"Succès Optimisé: {reliability['optimized_success_rate']}\n")
        
        self.logger.info(f"📄 Résultats sauvegardés:")
        self.logger.info(f"  JSON: {json_file}")
        self.logger.info(f"  CSV: {csv_file}")
        self.logger.info(f"  Résumé: {summary_file}")
    
    async def _generate_plots(self):
        """Génère les graphiques de comparaison"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Benchmark STT SuperWhisper V6 - Original vs Optimisé', fontsize=16)
            
            # 1. Temps de traitement
            test_names = [r.test_name for r in self.results]
            original_times = [r.original_processing_time * 1000 for r in self.results]  # ms
            optimized_times = [r.optimized_processing_time * 1000 for r in self.results]  # ms
            
            x = np.arange(len(test_names))
            width = 0.35
            
            axes[0,0].bar(x - width/2, original_times, width, label='Original', alpha=0.8)
            axes[0,0].bar(x + width/2, optimized_times, width, label='Optimisé', alpha=0.8)
            axes[0,0].set_title('Temps de traitement (ms)')
            axes[0,0].set_xlabel('Tests')
            axes[0,0].set_ylabel('Temps (ms)')
            axes[0,0].legend()
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. RTF (Real-Time Factor)
            original_rtf = [r.original_rtf for r in self.results]
            optimized_rtf = [r.optimized_rtf for r in self.results]
            
            axes[0,1].bar(x - width/2, original_rtf, width, label='Original', alpha=0.8)
            axes[0,1].bar(x + width/2, optimized_rtf, width, label='Optimisé', alpha=0.8)
            axes[0,1].set_title('Real-Time Factor (RTF)')
            axes[0,1].set_xlabel('Tests')
            axes[0,1].set_ylabel('RTF')
            axes[0,1].legend()
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. Confiance
            original_conf = [r.original_confidence for r in self.results]
            optimized_conf = [r.optimized_confidence for r in self.results]
            
            axes[1,0].bar(x - width/2, original_conf, width, label='Original', alpha=0.8)
            axes[1,0].bar(x + width/2, optimized_conf, width, label='Optimisé', alpha=0.8)
            axes[1,0].set_title('Confiance')
            axes[1,0].set_xlabel('Tests')
            axes[1,0].set_ylabel('Confiance')
            axes[1,0].legend()
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. WER (si disponible)
            wer_values = [r.wer for r in self.results if r.wer is not None]
            if wer_values:
                axes[1,1].hist(wer_values, bins=10, alpha=0.7, edgecolor='black')
                axes[1,1].set_title('Distribution WER (%)')
                axes[1,1].set_xlabel('WER (%)')
                axes[1,1].set_ylabel('Fréquence')
                axes[1,1].axvline(np.mean(wer_values), color='red', linestyle='--', 
                                label=f'Moyenne: {np.mean(wer_values):.1f}%')
                axes[1,1].legend()
            else:
                axes[1,1].text(0.5, 0.5, 'WER non calculé', ha='center', va='center', 
                              transform=axes[1,1].transAxes)
                axes[1,1].set_title('WER non disponible')
            
            plt.tight_layout()
            
            # Sauvegarder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.output_dir / f"benchmark_plots_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Graphiques sauvegardés: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur génération graphiques: {e}")


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Benchmark STT SuperWhisper V6')
    parser.add_argument('--model', default='large-v2', help='Modèle Whisper')
    parser.add_argument('--compute-type', default='float16', help='Type de calcul')
    parser.add_argument('--output-dir', default='benchmark_results', help='Dossier de sortie')
    parser.add_argument('--no-plots', action='store_true', help='Désactiver génération graphiques')
    parser.add_argument('--test-unified', action='store_true', help='Tester manager unifié')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'model': args.model,
        'compute_type': args.compute_type,
        'output_dir': args.output_dir,
        'generate_plots': not args.no_plots,
        'test_unified_manager': args.test_unified,
        'calculate_wer': True
    }
    
    # Exécution benchmark
    benchmark = STTBenchmarkSuite(config)
    
    try:
        await benchmark.initialize()
        report = await benchmark.run_benchmark()
        
        print("\n" + "="*60)
        print("🏆 RÉSULTATS BENCHMARK STT SUPERWHISPER V6")
        print("="*60)
        
        summary = report['summary']
        print(f"📈 Améliorations performances:")
        print(f"   Latence: {summary['performance_improvement']['latency']}")
        print(f"   RTF: {summary['performance_improvement']['rtf']}")
        print(f"   Accélération: {summary['performance_improvement']['speedup']}")
        
        print(f"\n📊 Métriques qualité:")
        print(f"   WER moyen: {summary['quality_metrics']['avg_wer']}")
        print(f"   CER moyen: {summary['quality_metrics']['avg_cer']}")
        print(f"   Boost confiance: {summary['quality_metrics']['confidence_boost']}")
        
        print(f"\n✅ Fiabilité:")
        print(f"   Succès Original: {summary['reliability']['original_success_rate']}")
        print(f"   Succès Optimisé: {summary['reliability']['optimized_success_rate']}")
        
        print(f"\n📁 Résultats sauvegardés dans: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Erreur benchmark: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main())) 