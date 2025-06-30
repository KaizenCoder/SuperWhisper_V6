#!/usr/bin/env python3
"""
Validateur de Configuration GPU - Luxa SuperWhisper V6 [VERSION RENFORCÉE]
==========================================================================
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Valide que tous les scripts du projet respectent les règles GPU obligatoires.
Basé sur les leçons du triple contrôle de sécurité GPU.

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

import re
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA non disponible - mode validation uniquement")
        return False
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    return True

class GPUConfigValidator:
    """Validateur de configuration GPU pour le projet [VERSION RENFORCÉE]"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations = []
        self.warnings = []
        self.files_checked = 0
        self.critical_files_found = {}
        
    def validate_project(self) -> Dict[str, Any]:
        """Valide tout le projet avec contrôles renforcés"""
        print(f"\n🔍 VALIDATION CONFIGURATION GPU RENFORCÉE - Projet: {self.project_root}")
        print("="*70)
        
        # Patterns critiques étendus basés sur l'audit triple contrôle
        critical_patterns = [
            # PATTERNS DANGEREUX ABSOLUS
            (r'device\s*=\s*["\']cuda:0["\']', "CUDA:0 (RTX 5060) explicite - INTERDIT"),
            (r'torch\.cuda\.set_device\s*\(\s*0\s*\)', "set_device(0) - RTX 5060 INTERDITE"),
            (r'\.to\s*\(\s*["\']cuda:0["\']\s*\)', ".to('cuda:0') - RTX 5060 INTERDITE"),
            (r'\.cuda\s*\(\s*0\s*\)', ".cuda(0) - RTX 5060 INTERDITE"),
            (r'selected_gpu\s*=\s*0', "selected_gpu = 0 - RTX 5060 INTERDITE"),
            (r'target_gpu\s*=\s*.*else\s+0', "Fallback vers GPU 0 - RTX 5060 INTERDITE"),
            (r'gpu_id\s*=\s*0', "gpu_id = 0 - RTX 5060 INTERDITE"),
            (r'device_id\s*=\s*0', "device_id = 0 - RTX 5060 INTERDITE"),
            (r'main_gpu\s*=\s*0', "main_gpu = 0 - configuration LLM interdite"),
            (r'get_device_name\s*\(\s*0\s*\)', "get_device_name(0) - référence RTX 5060"),
            (r'get_device_properties\s*\(\s*0\s*\)', "get_device_properties(0) - référence RTX 5060"),
            (r'get_device_capability\s*\(\s*0\s*\)', "get_device_capability(0) - référence RTX 5060"),
            (r'torch\.device\s*\(\s*["\']cuda:0["\']\s*\)', "torch.device('cuda:0') - RTX 5060 INTERDITE"),
        ]
        
        # Patterns obligatoires
        required_patterns = [
            (r'CUDA_VISIBLE_DEVICES.*1', "Configuration CUDA_VISIBLE_DEVICES"),
            (r'RTX 3090.*OBLIGATOIRE|CUDA:1.*OBLIGATOIRE', "Documentation RTX 3090"),
        ]
        
        # Scanner tous les fichiers Python
        python_files = list(self.project_root.rglob("*.py"))
        
        # Exclure les dossiers non pertinents
        excluded_dirs = {'venv_piper312', '__pycache__', '.git', 'node_modules'}
        python_files = [f for f in python_files if not any(excluded in f.parts for excluded in excluded_dirs)]
        
        print(f"📁 Fichiers Python trouvés: {len(python_files)}")
        
        for py_file in python_files:
            self._validate_python_file(py_file, critical_patterns, required_patterns)
        
        # Scanner fichiers PowerShell
        ps_files = list(self.project_root.rglob("*.ps1"))
        print(f"📁 Fichiers PowerShell trouvés: {len(ps_files)}")
        
        for ps_file in ps_files:
            self._validate_powershell_file(ps_file)
        
        # Scanner fichiers de configuration YAML/JSON
        config_files = list(self.project_root.rglob("*.yaml")) + list(self.project_root.rglob("*.yml")) + list(self.project_root.rglob("*.json"))
        config_files = [f for f in config_files if not any(excluded in f.parts for excluded in excluded_dirs)]
        print(f"📁 Fichiers de configuration trouvés: {len(config_files)}")
        
        for config_file in config_files:
            self._validate_config_file(config_file)
        
        # Valider fichiers de configuration spéciaux
        self._validate_special_files()
        
        # Valider fichiers critiques spécifiques
        self._validate_critical_files()
        
        return self._generate_report()
    
    def _has_cuda_visible_devices_1(self, content: str) -> bool:
        """Vérifie si le fichier configure CUDA_VISIBLE_DEVICES='1'"""
        return bool(re.search(r"os\.environ\[.*CUDA_VISIBLE_DEVICES.*\]\s*=\s*[\\'\\\"]1[\\'\\\"]\"\"", content))

    def _validate_python_file(self, file_path: Path, critical_patterns: List, required_patterns: List):
        """Valide un fichier Python avec contrôles renforcés"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            self.files_checked += 1
            
            # NE PAS ignorer les fichiers de test - ils sont critiques pour la sécurité
            is_test_file = 'test_' in file_path.name.lower()
            is_validator_file = 'validate_' in file_path.name.lower() or 'validator' in file_path.name.lower()
            is_benchmark_file = 'benchmark' in file_path.name.lower()
            
            # Vérifier si CUDA_VISIBLE_DEVICES='1' est configuré
            has_cuda_visible_1 = self._has_cuda_visible_devices_1(content)
            
            # Filtrer les commentaires historiques et docstrings
            content_to_check = self._filter_safe_content(content, file_path)
            
            # Détecter patterns critiques
            for pattern, description in critical_patterns:
                # Exclure certains patterns si CUDA_VISIBLE_DEVICES='1' est configuré
                if has_cuda_visible_1 and any(safe in description for safe in [
                    "Auto-sélection CUDA sans index", ".cuda() sans index"
                ]):
                    continue  # Pattern sécurisé avec CUDA_VISIBLE_DEVICES='1'
                    
                matches = re.finditer(pattern, content_to_check, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # Exclure les utilisations légitimes dans les fonctions de validation RTX 3090
                    if is_validator_file:
                        # Vérifier si le match est dans une fonction de validation RTX 3090
                        surrounding_lines = content.split('\n')[max(0, line_num-5):line_num+5]
                        if any('validate_rtx3090' in line or 'RTX 3090' in line for line in surrounding_lines):
                            continue
                    
                    # Récupérer la ligne complète pour le contexte
                    lines = content.split('\n')
                    if line_num <= len(lines):
                        line_content = lines[line_num-1].strip()
                        
                        violation = {
                            'file': str(file_path),
                            'line': line_num,
                            'pattern': pattern,
                            'description': description,
                            'code': match.group(),
                            'context': line_content
                        }
                        
                        # Catégoriser par type de fichier
                        if is_test_file:
                            violation['description'] = f"[FICHIER TEST] {description}"
                        elif is_benchmark_file:
                            violation['description'] = f"[FICHIER BENCHMARK] {description}"
                        
                        self.violations.append(violation)
            
            # Vérifier que les fichiers critiques ont les patterns requis
            # (Note: cette méthode pourrait être ajoutée plus tard si nécessaire)
                        
        except Exception as e:
            print(f"⚠️ Erreur lecture {file_path}: {e}")
    
    def _filter_safe_content(self, content: str, file_path: Path) -> str:
        """Filtre le contenu pour exclure les commentaires historiques et docstrings"""
        lines = content.split('\n')
        filtered_lines = []
        
        in_docstring = False
        docstring_delimiter = None
        
        for line in lines:
            original_line = line
            line_strip = line.strip()
            
            # Détecter début/fin de docstring
            if '"""' in line_strip or "'''" in line_strip:
                if not in_docstring:
                    # Début de docstring
                    in_docstring = True
                    docstring_delimiter = '"""' if '"""' in line_strip else "'''"
                elif docstring_delimiter in line_strip:
                    # Fin de docstring
                    in_docstring = False
                    docstring_delimiter = None
                    continue  # Exclure cette ligne aussi
                continue  # Exclure les lignes de docstring
            
            # Si on est dans une docstring, l'exclure
            if in_docstring:
                continue
                
            # Exclure les commentaires qui contiennent des marqueurs historiques
            if (line_strip.startswith('#') or line_strip.startswith('*')) and any(marker in line_strip for marker in [
                'AVANT:', 'HISTORIQUE', 'CORRIGÉ', 'OLD:', 'LEGACY', 'DEPRECATED'
            ]):
                continue
                
            # Exclure les commentaires dans les tests qui documentent l'ancien code
            if 'test_' in file_path.name.lower() and (
                'AVANT:' in line_strip or 
                'else 0' in line_strip and any(marker in line_strip for marker in ['❌', 'HISTORIQUE', 'CORRIGÉ'])
            ):
                continue
                
            # Exclure les patterns dans le validateur lui-même (exemples et patterns de test)
            if 'validate_gpu_config.py' in file_path.name and (
                'selected_gpu = 0' in line_strip or 
                'target_gpu = 1 if gpu_count >= 2 else 0' in line_strip or
                'gpu_count >= 2 else 0' in line_strip
            ):
                continue
                
            filtered_lines.append(original_line)
        
        return '\n'.join(filtered_lines)
    
    def _validate_config_file(self, file_path: Path):
        """Valide les fichiers de configuration YAML/JSON"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            self.files_checked += 1
            
            # Patterns critiques pour configuration
            config_patterns = [
                (r'gpu_device_index\s*:\s*0', "gpu_device_index: 0 dans config - RTX 5060 INTERDITE"),
                (r'gpu_device\s*:\s*["\']cuda:0["\']', "gpu_device: cuda:0 dans config - RTX 5060 INTERDITE"),
                (r'device_id\s*:\s*0', "device_id: 0 dans config - RTX 5060 INTERDITE"),
                (r'main_gpu\s*:\s*0', "main_gpu: 0 dans config - RTX 5060 INTERDITE"),
            ]
            
            for pattern, description in config_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.violations.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': line_num,
                        'type': 'CRITIQUE',
                        'description': description,
                        'code': match.group(0).strip()
                    })
            
            # Vérifier configurations positives
            if 'gpu_device_index' in content and 'gpu_device_index: 1' not in content:
                self.warnings.append({
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': 1,
                    'type': 'AVERTISSEMENT',
                    'description': 'Fichier config avec gpu_device_index mais pas fixé à 1',
                    'code': 'Vérifier gpu_device_index: 1'
                })
                
        except Exception as e:
            self.warnings.append({
                'file': str(file_path.relative_to(self.project_root)),
                'line': 0,
                'type': 'ERREUR',
                'description': f'Erreur lecture fichier config: {e}',
                'code': ''
            })
    
    def _validate_powershell_file(self, file_path: Path):
        """Valide un fichier PowerShell"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            self.files_checked += 1
            
            # Vérifier configuration GPU PowerShell
            if '$env:CUDA_VISIBLE_DEVICES' not in content:
                self.warnings.append({
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': 1,
                    'type': 'AVERTISSEMENT',
                    'description': 'Script PowerShell sans configuration CUDA_VISIBLE_DEVICES',
                    'code': 'Ajouter $env:CUDA_VISIBLE_DEVICES = "1"'
                })
                
        except Exception as e:
            self.warnings.append({
                'file': str(file_path.relative_to(self.project_root)),
                'line': 0,
                'type': 'ERREUR',
                'description': f'Erreur lecture fichier PowerShell: {e}',
                'code': ''
            })
    
    def _validate_special_files(self):
        """Valide les fichiers de configuration spéciaux"""
        
        # Vérifier gpu_manager.py
        gpu_manager_file = self.project_root / "utils" / "gpu_manager.py"
        if gpu_manager_file.exists():
            content = gpu_manager_file.read_text(encoding='utf-8', errors='ignore')
            self.critical_files_found['gpu_manager'] = True
            
            # Vérifier que les fallbacks pointent vers CUDA:1
            if 'cuda:0' in content and 'RTX 5060' not in content:
                self.violations.append({
                    'file': 'utils/gpu_manager.py',
                    'line': content.find('cuda:0') + 1,
                    'type': 'CRITIQUE',
                    'description': 'gpu_manager contient cuda:0 sans documentation RTX 5060',
                    'code': 'cuda:0 trouvé'
                })
        
        # Vérifier stt_manager_robust.py
        stt_manager_file = self.project_root / "STT" / "stt_manager_robust.py"
        if stt_manager_file.exists():
            content = stt_manager_file.read_text(encoding='utf-8', errors='ignore')
            self.critical_files_found['stt_manager'] = True
            
            # Vérifier que selected_gpu = 1
            if 'selected_gpu = 0' in content:
                self.violations.append({
                    'file': 'STT/stt_manager_robust.py',
                    'line': content.find('selected_gpu = 0') + 1,
                    'type': 'CRITIQUE',
                    'description': 'selected_gpu = 0 trouvé - RTX 5060 interdite',
                    'code': 'selected_gpu = 0'
                })
        
        # Vérifier llm_manager_enhanced.py
        llm_manager_file = self.project_root / "LLM" / "llm_manager_enhanced.py"
        if llm_manager_file.exists():
            content = llm_manager_file.read_text(encoding='utf-8', errors='ignore')
            self.critical_files_found['llm_manager'] = True
            
            # Vérifier protection contre gpu_device_index=0
            if 'gpu_device_index == 0' not in content:
                self.warnings.append({
                    'file': 'LLM/llm_manager_enhanced.py',
                    'line': 1,
                    'type': 'RECOMMANDATION',
                    'description': 'LLM Manager sans protection explicite contre gpu_device_index=0',
                    'code': 'Ajouter validation gpu_device_index != 0'
                })
    
    def _validate_critical_files(self):
        """Valide les fichiers critiques identifiés pendant l'audit"""
        
        critical_test_files = [
            "tests/test_stt_handler.py",
            "tests/test_llm_handler.py",
            "tests/test_enhanced_llm_manager.py",
            "test_tts_rtx3090_performance.py",
            "test_rtx3090_detection.py"
        ]
        
        for critical_file in critical_test_files:
            file_path = self.project_root / critical_file
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                self.critical_files_found[critical_file] = True
                
                # Vérifier patterns spécifiques de test
                if 'device="cuda:0"' in content:
                    self.violations.append({
                        'file': critical_file,
                        'line': content.find('device="cuda:0"') + 1,
                        'type': 'CRITIQUE',
                        'description': 'Fichier test critique utilise device="cuda:0"',
                        'code': 'device="cuda:0"'
                    })
    
    def _generate_report(self) -> Dict[str, Any]:
        """Génère le rapport de validation renforcé"""
        
        critical_count = len(self.violations)
        warning_count = len(self.warnings)
        
        report = {
            'files_checked': self.files_checked,
            'critical_violations': critical_count,
            'warnings': warning_count,
            'status': 'ÉCHEC' if critical_count > 0 else 'SUCCÈS',
            'violations': self.violations,
            'warnings_list': self.warnings,
            'critical_files_found': self.critical_files_found,
            'validation_timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Affiche le rapport de validation renforcé"""
        
        print(f"\n📊 RAPPORT DE VALIDATION GPU RENFORCÉE")
        print("="*70)
        print(f"⏰ Timestamp: {report['validation_timestamp']}")
        print(f"📁 Fichiers vérifiés: {report['files_checked']}")
        print(f"🚨 Violations critiques: {report['critical_violations']}")
        print(f"⚠️ Avertissements: {report['warnings']}")
        print(f"🎯 Statut global: {report['status']}")
        
        # Afficher fichiers critiques trouvés
        if report['critical_files_found']:
            print(f"\n🔍 FICHIERS CRITIQUES ANALYSÉS:")
            print("-" * 50)
            for file_key, found in report['critical_files_found'].items():
                status = "✅ TROUVÉ" if found else "❌ MANQUANT"
                print(f"   {file_key}: {status}")
        
        if report['violations']:
            print(f"\n🚨 VIOLATIONS CRITIQUES ({len(report['violations'])}):")
            print("-" * 50)
            for violation in report['violations']:
                print(f"📄 {violation['file']}:{violation['line']}")
                print(f"   🔴 {violation['description']}")
                print(f"   💻 Code: {violation['code']}")
                print()
        
        if report['warnings_list']:
            print(f"\n⚠️ AVERTISSEMENTS ({len(report['warnings_list'])}):")
            print("-" * 50)
            for warning in report['warnings_list'][:10]:  # Limiter à 10
                print(f"📄 {warning['file']}:{warning['line']}")
                print(f"   🟡 {warning['description']}")
                print()
            
            if len(report['warnings_list']) > 10:
                print(f"   ... et {len(report['warnings_list']) - 10} autres avertissements")
        
        # Recommandations renforcées
        print(f"\n💡 RECOMMANDATIONS RENFORCÉES:")
        print("-" * 50)
        print("1. Tous les scripts Python doivent commencer par:")
        print("   os.environ['CUDA_VISIBLE_DEVICES'] = '1'")
        print("2. Utiliser 'cuda:1' ou torch.device('cuda:1') exclusivement")
        print("3. Ajouter validate_rtx3090_configuration() dans les scripts principaux")
        print("4. Documenter la configuration dual-GPU dans les commentaires")
        print("5. Fichiers de test DOIVENT utiliser device='cuda:1' explicitement")
        print("6. Configurations YAML/JSON: gpu_device_index: 1, gpu_device: 'cuda:1'")
        print("7. Validation obligatoire VRAM 24GB pour signature RTX 3090")
        
        if report['status'] == 'SUCCÈS':
            print(f"\n🎉 VALIDATION RÉUSSIE - Configuration GPU conforme!")
            print(f"🔒 Sécurité RTX 3090 exclusive garantie")
        else:
            print(f"\n🚫 VALIDATION ÉCHOUÉE - {report['critical_violations']} violations critiques à corriger")
            print(f"🛑 DÉVELOPPEMENT BLOQUÉ jusqu'à correction complète")

def main():
    """Fonction principale de validation renforcée"""
    
    # Valider d'abord notre propre configuration
    print("🔧 INITIALISATION VALIDATEUR GPU RENFORCÉ")
    print("="*50)
    validate_rtx3090_configuration()
    
    # Lancer la validation du projet
    project_root = Path(__file__).parent
    validator = GPUConfigValidator(project_root)
    
    report = validator.validate_project()
    validator.print_report(report)
    
    # Sauvegarder rapport détaillé
    report_file = project_root / "docs" / "phase_1" / "validation_gpu_report.json"
    try:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Rapport sauvegardé: {report_file}")
    except Exception as e:
        print(f"\n⚠️ Impossible de sauvegarder le rapport: {e}")
    
    # Code de sortie
    exit_code = 1 if report['critical_violations'] > 0 else 0
    
    if exit_code == 0:
        print(f"\n✅ Validation terminée - Aucune violation critique")
        print(f"🚀 DÉVELOPPEMENT AUTORISÉ avec sécurité RTX 3090")
    else:
        print(f"\n❌ Validation échouée - Corriger les violations critiques")
        print(f"🛑 DÉVELOPPEMENT BLOQUÉ - Configuration GPU non sécurisée")
    
    return exit_code

if __name__ == "__main__":
    # 🚨 CRITIQUE: Configuration dual-GPU RTX 5060 (CUDA:0) + RTX 3090 (CUDA:1)
    # RTX 5060 (CUDA:0) = INTERDITE (8GB insuffisant)
    # RTX 3090 (CUDA:1) = SEULE AUTORISÉE (24GB VRAM)
    
    exit_code = main()
    sys.exit(exit_code)
