#!/usr/bin/env python3
"""
🏆 VALIDATEUR COMPLET SUPERWHISPER V6 - MISSION GPU RTX 3090
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Teste tous les modules du projet SuperWhisper V6 et indique leur statut fonctionnel
après homogénéisation GPU RTX 3090.
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

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
import json
from datetime import datetime

def validate_rtx3090_mandatory():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER', '')
    if cuda_order != 'PCI_BUS_ID':
        raise RuntimeError(f"🚫 CUDA_DEVICE_ORDER='{cuda_order}' incorrect - doit être 'PCI_BUS_ID'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

class SuperWhisperValidator:
    """Validateur complet des modules SuperWhisper V6"""
    
    def __init__(self):
        self.results = {}
        self.project_root = Path.cwd()
        
        # Liste des modules critiques à tester (Phase 1 + Phase 2)
        self.modules_to_test = [
            # Phase 1 - Scripts principaux (6 modules)
            ("superwhisper_v6.py", "Script principal SuperWhisper V6", True),
            ("audio_input_handler.py", "Gestionnaire d'entrée audio", True),
            ("luxa_superwhisper_v6_gui.py", "Interface graphique GUI", False),
            ("superwhisper_v6_debug.py", "Mode debug SuperWhisper", False),
            ("test_memory_leak_v4.py", "Test Memory Leak V4", False),
            ("memory_leak_v4.py", "Memory Leak Manager V4", False),
            
            # Phase 2 - Modules Core (13 modules)
            ("STT/stt_manager_optimized.py", "Manager STT optimisé", True),
            ("STT/transcription_with_context.py", "Transcription avec contexte", True),
            ("STT/vad_manager_optimized.py", "VAD Manager optimisé", True),
            ("STT/whisper_manager_gpu.py", "Whisper Manager GPU", True),
            ("LUXA_LLM/claude_handler.py", "Handler Claude", False),
            ("LUXA_LLM/llm_luxa_handler.py", "LLM LUXA Handler", False),
            ("TTS/tts_handler_coqui.py", "TTS Handler Coqui", True),
            ("TTS/tts_handler_piper_native.py", "TTS Handler Piper Native", True),
            ("LLM/llm_manager_enhanced.py", "LLM Manager Enhanced", True),
            ("LUXA_TTS/tts_handler_coqui.py", "LUXA TTS Handler Coqui", True),
            ("Orchestrator/fallback_manager.py", "Fallback Manager", False),
            ("benchmarks/benchmark_stt_realistic.py", "Benchmark STT réaliste", False),
        ]
    
    def check_gpu_configuration_in_file(self, filepath: Path) -> Tuple[bool, str]:
        """Vérifie si un fichier a la configuration GPU RTX 3090 correcte"""
        try:
            if not filepath.exists():
                return False, f"Fichier introuvable: {filepath}"
            
            content = filepath.read_text(encoding='utf-8')
            
            # Vérifications critiques
            checks = [
                ("CUDA_VISIBLE_DEVICES", "os.environ['CUDA_VISIBLE_DEVICES'] = '1'", "Forçage RTX 3090"),
                ("CUDA_DEVICE_ORDER", "os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'", "Ordre GPU stable"),
                ("PYTORCH_CUDA_ALLOC_CONF", "os.environ['PYTORCH_CUDA_ALLOC_CONF']", "Optimisation mémoire"),
                ("validate_rtx3090", "def validate_rtx3090", "Fonction validation obligatoire"),
                ("Shebang", "#!/usr/bin/env python3", "Shebang standard"),
            ]
            
            missing = []
            for name, pattern, desc in checks:
                if pattern not in content:
                    missing.append(f"{name} ({desc})")
            
            if missing:
                return False, f"Configuration GPU incomplète: {', '.join(missing)}"
            
            return True, "Configuration GPU RTX 3090 complète"
            
        except Exception as e:
            return False, f"Erreur lecture fichier: {str(e)}"
    
    def test_module_import(self, filepath: Path) -> Tuple[bool, str]:
        """Teste si un module peut être importé sans erreur"""
        try:
            # Convertir chemin en nom de module
            module_name = str(filepath.with_suffix(''))
            module_name = module_name.replace('/', '.').replace('\\', '.')
            if module_name.startswith('.'):
                module_name = module_name[1:]
            
            # Charger le module
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None:
                return False, "Impossible de créer spec module"
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return True, "Import réussi"
            
        except ImportError as e:
            return False, f"Dépendance manquante: {str(e)}"
        except Exception as e:
            return False, f"Erreur import: {type(e).__name__}: {str(e)}"
    
    def test_module(self, filename: str, description: str, gpu_required: bool) -> Dict:
        """Teste un module spécifique"""
        print(f"\n🔍 Test: {filename}")
        print(f"   📝 {description}")
        
        filepath = self.project_root / filename
        result = {
            'filename': filename,
            'description': description,
            'gpu_required': gpu_required,
            'file_exists': False,
            'gpu_config_ok': False,
            'import_ok': False,
            'functional': False,
            'errors': [],
            'warnings': []
        }
        
        # Test 1: Existence du fichier
        if not filepath.exists():
            result['errors'].append(f"Fichier introuvable: {filepath}")
            print(f"   ❌ Fichier introuvable")
            return result
        
        result['file_exists'] = True
        print(f"   ✅ Fichier existe")
        
        # Test 2: Configuration GPU (si requise)
        if gpu_required:
            gpu_ok, gpu_msg = self.check_gpu_configuration_in_file(filepath)
            result['gpu_config_ok'] = gpu_ok
            if gpu_ok:
                print(f"   ✅ Config GPU: {gpu_msg}")
            else:
                result['errors'].append(f"Config GPU: {gpu_msg}")
                print(f"   ❌ Config GPU: {gpu_msg}")
        else:
            result['gpu_config_ok'] = True  # Non applicable
            result['warnings'].append("Module sans GPU - config non vérifiée")
            print(f"   ⚠️  GPU non requis pour ce module")
        
        # Test 3: Import du module
        import_ok, import_msg = self.test_module_import(filepath)
        result['import_ok'] = import_ok
        if import_ok:
            print(f"   ✅ Import: {import_msg}")
        else:
            result['errors'].append(f"Import: {import_msg}")
            print(f"   ❌ Import: {import_msg}")
        
        # Déterminer si fonctionnel
        result['functional'] = result['file_exists'] and result['gpu_config_ok'] and result['import_ok']
        
        status = "✅ FONCTIONNEL" if result['functional'] else "❌ NON FONCTIONNEL"
        print(f"   🏆 Statut: {status}")
        
        return result
    
    def run_validation(self) -> Dict:
        """Lance la validation complète de tous les modules"""
        print("🏆 VALIDATION COMPLÈTE SUPERWHISPER V6 - MISSION GPU RTX 3090")
        print("=" * 70)
        print(f"📂 Projet: {self.project_root}")
        print(f"🎮 GPU: RTX 3090 (CUDA:1) exclusif")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Validation GPU système
        try:
            validate_rtx3090_mandatory()
        except Exception as e:
            print(f"\n🚨 ERREUR CRITIQUE GPU: {e}")
            return {'error': str(e)}
        
        # Test de tous les modules
        for filename, description, gpu_required in self.modules_to_test:
            result = self.test_module(filename, description, gpu_required)
            self.results[filename] = result
        
        return self.generate_summary()
    
    def generate_summary(self) -> Dict:
        """Génère un résumé complet de la validation"""
        print("\n" + "=" * 70)
        print("📋 RÉSUMÉ VALIDATION SUPERWHISPER V6")
        print("=" * 70)
        
        # Statistiques
        total = len(self.results)
        functional = sum(1 for r in self.results.values() if r['functional'])
        non_functional = total - functional
        
        gpu_required = sum(1 for r in self.results.values() if r['gpu_required'])
        gpu_ok = sum(1 for r in self.results.values() if r['gpu_required'] and r['gpu_config_ok'])
        
        print(f"📊 STATISTIQUES:")
        print(f"   • Total modules testés: {total}")
        print(f"   • ✅ Fonctionnels: {functional} ({functional/total*100:.1f}%)")
        print(f"   • ❌ Non fonctionnels: {non_functional} ({non_functional/total*100:.1f}%)")
        print(f"   • 🎮 Modules GPU requis: {gpu_required}")
        print(f"   • 🎮 Config GPU OK: {gpu_ok}/{gpu_required}")
        
        # Détail par statut
        print(f"\n✅ MODULES FONCTIONNELS ({functional}):")
        for filename, result in self.results.items():
            if result['functional']:
                gpu_mark = "🎮" if result['gpu_required'] else "💻"
                print(f"   {gpu_mark} {filename} - {result['description']}")
        
        if non_functional > 0:
            print(f"\n❌ MODULES NON FONCTIONNELS ({non_functional}):")
            for filename, result in self.results.items():
                if not result['functional']:
                    gpu_mark = "🎮" if result['gpu_required'] else "💻"
                    print(f"   {gpu_mark} {filename} - {result['description']}")
                    for error in result['errors']:
                        print(f"      🔸 {error}")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if non_functional == 0:
            print("   🎉 Tous les modules sont fonctionnels !")
            print("   🚀 SuperWhisper V6 prêt pour la production")
        else:
            print(f"   🔧 Corriger {non_functional} module(s) non fonctionnel(s)")
            print("   🎮 Vérifier configurations GPU manquantes")
            print("   📦 Installer dépendances manquantes")
        
        # Résumé mission GPU
        gpu_success_rate = gpu_ok / gpu_required * 100 if gpu_required > 0 else 100
        print(f"\n🏆 MISSION GPU RTX 3090:")
        print(f"   • Taux de succès: {gpu_success_rate:.1f}%")
        if gpu_success_rate == 100:
            print("   ✅ Mission GPU RÉUSSIE - RTX 3090 exclusive")
        else:
            print("   ⚠️  Mission GPU EN COURS - modules à corriger")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_modules': total,
            'functional_modules': functional,
            'non_functional_modules': non_functional,
            'gpu_modules_required': gpu_required,
            'gpu_modules_ok': gpu_ok,
            'gpu_success_rate': gpu_success_rate,
            'mission_status': 'COMPLETED' if gpu_success_rate == 100 else 'IN_PROGRESS',
            'results': self.results
        }
    
    def export_report(self, filepath: str):
        """Exporte le rapport en JSON"""
        summary = self.generate_summary()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Rapport exporté: {filepath}")

def main():
    """Fonction principale de validation"""
    validator = SuperWhisperValidator()
    
    try:
        summary = validator.run_validation()
        
        # Export rapport
        report_path = "validation_report_superwhisper_v6.json"
        validator.export_report(report_path)
        
        # Code de sortie
        if summary.get('gpu_success_rate', 0) == 100:
            print("\n🎉 VALIDATION RÉUSSIE - SuperWhisper V6 opérationnel")
            sys.exit(0)
        else:
            print("\n⚠️  VALIDATION PARTIELLE - corrections nécessaires")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n🚨 ERREUR VALIDATION: {e}")
        traceback.print_exc()
        sys.exit(2)

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    main() 