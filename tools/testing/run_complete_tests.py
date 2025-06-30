#!/usr/bin/env python3
"""
Script d'Exécution Complète des Tests TTS - SuperWhisper V6
Orchestration de tous les tests : pytest, démonstration, monitoring
🧪 Suite complète de validation Phase 3

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

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

class TTSTestSuite:
    """Suite complète de tests TTS avec orchestration et rapports"""
    
    def __init__(self):
        self.start_time = time.perf_counter()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'gpu_config': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'tests': {}
        }
        
    def log(self, message, level="INFO"):
        """Logging avec timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_command(self, command, description, timeout=300):
        """Exécution d'une commande avec capture des résultats"""
        self.log(f"🔄 {description}")
        self.log(f"   Commande: {' '.join(command)}")
        
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            
            execution_time = time.perf_counter() - start_time
            
            success = result.returncode == 0
            
            test_result = {
                'success': success,
                'execution_time': execution_time,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(command)
            }
            
            if success:
                self.log(f"✅ {description} réussi ({execution_time:.1f}s)")
            else:
                self.log(f"❌ {description} échoué ({execution_time:.1f}s)")
                self.log(f"   Code retour: {result.returncode}")
                if result.stderr:
                    self.log(f"   Erreur: {result.stderr[:200]}...")
                    
            return test_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.perf_counter() - start_time
            self.log(f"⏰ {description} timeout après {execution_time:.1f}s")
            return {
                'success': False,
                'execution_time': execution_time,
                'error': 'timeout',
                'command': ' '.join(command)
            }
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self.log(f"💥 {description} erreur: {e}")
            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'command': ' '.join(command)
            }
    
    def run_pytest_tests(self):
        """Exécution des tests pytest"""
        self.log("🧪 PHASE 1: Tests Pytest d'Intégration", "PHASE")
        
        # Vérification de l'existence du fichier de test
        test_file = Path("tests/test_tts_manager_integration.py")
        if not test_file.exists():
            self.log("❌ Fichier de test pytest non trouvé", "ERROR")
            return {
                'success': False,
                'error': 'Test file not found',
                'execution_time': 0
            }
        
        # Exécution des tests pytest
        command = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v", "-s",
            "--tb=short",
            "--disable-warnings",
            "--color=yes"
        ]
        
        return self.run_command(command, "Tests Pytest d'Intégration", timeout=600)
    
    def run_demo_batch(self):
        """Exécution de la démonstration batch"""
        self.log("🎵 PHASE 2: Démonstration Batch", "PHASE")
        
        demo_file = Path("scripts/demo_tts.py")
        if not demo_file.exists():
            self.log("❌ Script de démonstration non trouvé", "ERROR")
            return {
                'success': False,
                'error': 'Demo script not found',
                'execution_time': 0
            }
        
        # Simulation d'entrée pour le mode batch (choix 2)
        command = [sys.executable, str(demo_file)]
        
        # Pour automatiser, on utilise echo pour simuler l'entrée
        if os.name == 'nt':  # Windows
            full_command = f'echo 2 | python "{demo_file}"'
            result = self.run_command(["cmd", "/c", full_command], "Démonstration Batch TTS", timeout=300)
        else:  # Unix/Linux
            full_command = f'echo "2" | python "{demo_file}"'
            result = self.run_command(["bash", "-c", full_command], "Démonstration Batch TTS", timeout=300)
        
        return result
    
    def run_performance_tests(self):
        """Exécution des tests de performance existants"""
        self.log("⚡ PHASE 3: Tests de Performance", "PHASE")
        
        performance_tests = [
            "test_performance_simple.py",
            "monitor_phase3_demo.py"
        ]
        
        results = {}
        
        for test_script in performance_tests:
            test_path = Path(test_script)
            if test_path.exists():
                command = [sys.executable, str(test_path)]
                result = self.run_command(command, f"Test Performance {test_script}", timeout=180)
                results[test_script] = result
            else:
                self.log(f"⚠️ Script {test_script} non trouvé", "WARNING")
                results[test_script] = {
                    'success': False,
                    'error': 'Script not found',
                    'execution_time': 0
                }
        
        return results
    
    def check_system_requirements(self):
        """Vérification des prérequis système"""
        self.log("🔍 PHASE 0: Vérification des Prérequis", "PHASE")
        
        checks = {}
        
        # Vérification Python
        python_version = sys.version_info
        checks['python_version'] = {
            'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            'valid': python_version >= (3, 8)
        }
        
        # Vérification des modules critiques
        critical_modules = ['torch', 'yaml', 'pytest', 'asyncio']
        for module in critical_modules:
            try:
                __import__(module)
                checks[f'module_{module}'] = {'available': True}
                self.log(f"✅ Module {module} disponible")
            except ImportError:
                checks[f'module_{module}'] = {'available': False}
                self.log(f"❌ Module {module} manquant", "ERROR")
        
        # Vérification GPU
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                checks['gpu'] = {
                    'available': True,
                    'name': gpu_name,
                    'memory_gb': gpu_memory,
                    'cuda_device': os.environ.get('CUDA_VISIBLE_DEVICES')
                }
                self.log(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                checks['gpu'] = {'available': False}
                self.log("❌ GPU CUDA non disponible", "ERROR")
        except Exception as e:
            checks['gpu'] = {'available': False, 'error': str(e)}
            self.log(f"❌ Erreur vérification GPU: {e}", "ERROR")
        
        # Vérification fichiers de configuration
        config_files = ['config/tts.yaml']
        for config_file in config_files:
            config_path = Path(config_file)
            checks[f'config_{config_file}'] = {
                'exists': config_path.exists(),
                'path': str(config_path)
            }
            if config_path.exists():
                self.log(f"✅ Configuration {config_file} trouvée")
            else:
                self.log(f"❌ Configuration {config_file} manquante", "ERROR")
        
        return checks
    
    def generate_report(self):
        """Génération du rapport final"""
        total_time = time.perf_counter() - self.start_time
        
        self.log("📊 GÉNÉRATION DU RAPPORT FINAL", "PHASE")
        
        # Calcul des statistiques
        all_tests = []
        for phase, tests in self.results['tests'].items():
            if isinstance(tests, dict):
                if 'success' in tests:  # Test unique
                    all_tests.append(tests)
                else:  # Groupe de tests
                    for test_name, test_result in tests.items():
                        if isinstance(test_result, dict) and 'success' in test_result:
                            all_tests.append(test_result)
        
        successful_tests = [t for t in all_tests if t.get('success', False)]
        failed_tests = [t for t in all_tests if not t.get('success', False)]
        
        total_execution_time = sum(t.get('execution_time', 0) for t in all_tests)
        
        # Rapport de synthèse
        report = {
            'summary': {
                'total_duration': total_time,
                'total_tests': len(all_tests),
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(successful_tests) / len(all_tests) if all_tests else 0,
                'total_execution_time': total_execution_time
            },
            'details': self.results
        }
        
        # Sauvegarde du rapport
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_complete_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Affichage du rapport
        print("\n" + "="*80)
        print("📊 RAPPORT FINAL DES TESTS TTS SUPERWHISPER V6")
        print("="*80)
        print(f"🕒 Durée totale: {total_time:.1f}s")
        print(f"🧪 Tests exécutés: {len(all_tests)}")
        print(f"✅ Réussis: {len(successful_tests)}")
        print(f"❌ Échecs: {len(failed_tests)}")
        print(f"📈 Taux de réussite: {len(successful_tests)/len(all_tests)*100:.1f}%" if all_tests else "📈 Aucun test exécuté")
        print(f"💾 Rapport détaillé: {report_file}")
        
        if failed_tests:
            print(f"\n❌ TESTS ÉCHOUÉS:")
            for i, test in enumerate(failed_tests, 1):
                print(f"   {i}. {test.get('command', 'Commande inconnue')}")
                if 'error' in test:
                    print(f"      Erreur: {test['error']}")
        
        print("\n" + "="*80)
        
        return report
    
    def run_all_tests(self):
        """Exécution de tous les tests"""
        self.log("🚀 DÉBUT DE LA SUITE COMPLÈTE DE TESTS TTS", "START")
        
        try:
            # Phase 0: Prérequis
            self.results['tests']['system_requirements'] = self.check_system_requirements()
            
            # Phase 1: Tests Pytest
            self.results['tests']['pytest_integration'] = self.run_pytest_tests()
            
            # Phase 2: Démonstration Batch
            self.results['tests']['demo_batch'] = self.run_demo_batch()
            
            # Phase 3: Tests de Performance
            self.results['tests']['performance_tests'] = self.run_performance_tests()
            
            # Génération du rapport final
            report = self.generate_report()
            
            # Détermination du statut global
            success_rate = report['summary']['success_rate']
            if success_rate >= 0.8:
                self.log("🎉 SUITE DE TESTS GLOBALEMENT RÉUSSIE", "SUCCESS")
                return True
            else:
                self.log(f"⚠️ SUITE DE TESTS PARTIELLEMENT RÉUSSIE ({success_rate:.1%})", "WARNING")
                return False
                
        except Exception as e:
            self.log(f"💥 ERREUR CRITIQUE DANS LA SUITE DE TESTS: {e}", "CRITICAL")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Point d'entrée principal"""
    print("🧪 SuperWhisper V6 - Suite Complète de Tests TTS")
    print("🚀 Validation Phase 3 : Pytest + Démonstration + Performance")
    print()
    
    # Vérification des arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("Usage: python run_complete_tests.py [--help]")
            print()
            print("Options:")
            print("  --help    Affiche cette aide")
            print()
            print("Ce script exécute automatiquement:")
            print("  1. Vérification des prérequis système")
            print("  2. Tests pytest d'intégration")
            print("  3. Démonstration batch TTS")
            print("  4. Tests de performance")
            print("  5. Génération du rapport final")
            return
    
    # Exécution de la suite de tests
    test_suite = TTSTestSuite()
    success = test_suite.run_all_tests()
    
    # Code de sortie
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 