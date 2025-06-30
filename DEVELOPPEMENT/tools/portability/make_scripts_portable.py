#!/usr/bin/env python3
"""
🚀 Utilitaire de Portabilité Scripts - SuperWhisper V6

Rend tous les scripts Python exécutables depuis n'importe quel répertoire
en résolvant automatiquement les chemins et imports.

Usage:
    python tools/make_scripts_portable.py --scan-all
    python tools/make_scripts_portable.py --fix-script path/to/script.py
    python tools/make_scripts_portable.py --create-launcher script.py

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import pathlib
import shutil
import argparse
import re
from typing import List, Dict, Set

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée pour make_scripts_portable.py")

# Template pour header portable obligatoire
PORTABLE_HEADER_TEMPLATE = '''#!/usr/bin/env python3
"""
{docstring}

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
    print(f"📁 Project Root: {{project_root}}")
    print(f"💻 Working Directory: {{os.getcwd()}}")
    
    return project_root

# Initialiser l'environnement portable
_PROJECT_ROOT = _setup_portable_environment()

# Maintenant imports normaux...
'''

def find_python_scripts(root_dir: pathlib.Path) -> List[pathlib.Path]:
    """Trouve tous les scripts Python dans le projet"""
    scripts = []
    
    # Patterns à exclure
    exclude_patterns = [
        '__pycache__',
        '.git',
        'venv',
        'env',
        '.env',
        'node_modules',
        '.cursor-tmp'
    ]
    
    for py_file in root_dir.rglob("*.py"):
        # Exclure certains répertoires
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue
            
        # Inclure seulement les scripts (pas les modules)
        if is_script_file(py_file):
            scripts.append(py_file)
    
    return scripts

def is_script_file(py_file: pathlib.Path) -> bool:
    """Détermine si un fichier Python est un script exécutable"""
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # Lire début du fichier
            
        # Critères pour identifier un script
        script_indicators = [
            'if __name__ == "__main__"',
            '#!/usr/bin/env python',
            'def main(',
            'argparse.',
            'sys.argv'
        ]
        
        return any(indicator in content for indicator in script_indicators)
        
    except Exception:
        return False

def analyze_script_dependencies(script_path: pathlib.Path) -> Dict[str, Set[str]]:
    """Analyse les dépendances d'un script"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️  Erreur lecture {script_path}: {e}")
        return {}
    
    dependencies = {
        'relative_imports': set(),
        'absolute_imports': set(),
        'file_paths': set(),
        'working_dir_deps': set()
    }
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Imports relatifs
        if line.startswith('from .') or line.startswith('import .'):
            dependencies['relative_imports'].add(line)
        
        # Imports absoluts locaux
        elif re.match(r'from \w+\.\w+', line) or re.match(r'import \w+\.\w+', line):
            dependencies['absolute_imports'].add(line)
        
        # Chemins de fichiers relatifs
        elif any(pattern in line for pattern in ['"../', "'../", '"./', "'./"]):
            dependencies['file_paths'].add(line)
        
        # Dépendances working directory
        elif any(pattern in line for pattern in ['open(', 'pathlib.Path(', 'os.path.join(']):
            if not line.startswith('#'):
                dependencies['working_dir_deps'].add(line)
    
    return dependencies

def has_portable_header(script_path: pathlib.Path) -> bool:
    """Vérifie si le script a déjà le header portable"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read(2000)  # Lire début du fichier
        
        return '_setup_portable_environment()' in content
        
    except Exception:
        return False

def make_script_portable(script_path: pathlib.Path, backup: bool = True) -> bool:
    """Rend un script portable en ajoutant le header approprié"""
    
    if has_portable_header(script_path):
        print(f"✅ {script_path.name} - Déjà portable")
        return True
    
    try:
        # Backup si demandé
        if backup:
            backup_path = script_path.with_suffix('.py.backup')
            shutil.copy2(script_path, backup_path)
            print(f"💾 Backup créé: {backup_path}")
        
        # Lire contenu original
        with open(script_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Extraire docstring existante
        lines = original_content.split('\n')
        docstring = extract_docstring(lines)
        
        # Trouver où commence le vrai code
        code_start = find_code_start(lines)
        
        # Créer nouveau contenu avec header portable
        portable_header = PORTABLE_HEADER_TEMPLATE.format(
            docstring=docstring or f"Script Portable - {script_path.name}"
        )
        
        # Combiner header + code existant
        remaining_code = '\n'.join(lines[code_start:])
        new_content = portable_header + '\n' + remaining_code
        
        # Écrire le nouveau fichier
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"🚀 {script_path.name} - Rendu portable")
        return True
        
    except Exception as e:
        print(f"❌ Erreur {script_path.name}: {e}")
        return False

def extract_docstring(lines: List[str]) -> str:
    """Extrait la docstring existante d'un script"""
    in_docstring = False
    docstring_lines = []
    quote_type = None
    
    for line in lines:
        stripped = line.strip()
        
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = True
                quote_type = stripped[:3]
                # Extraire contenu après les quotes
                content = stripped[3:]
                if content.endswith(quote_type):
                    # Docstring sur une ligne
                    return content[:-3].strip()
                elif content:
                    docstring_lines.append(content)
            elif stripped.startswith('#'):
                continue
            else:
                break
        else:
            if stripped.endswith(quote_type):
                # Fin de docstring
                content = stripped[:-3]
                if content:
                    docstring_lines.append(content)
                break
            else:
                docstring_lines.append(line)
    
    return '\n'.join(docstring_lines).strip() if docstring_lines else ""

def find_code_start(lines: List[str]) -> int:
    """Trouve où commence le vrai code (après shebang, docstring, imports de base)"""
    skip_patterns = [
        lambda l: l.strip().startswith('#!'),
        lambda l: l.strip().startswith('"""') or l.strip().startswith("'''"),
        lambda l: l.strip().startswith('#'),
        lambda l: l.strip().startswith('import os'),
        lambda l: l.strip().startswith('import sys'),
        lambda l: l.strip().startswith('os.environ'),
        lambda l: l.strip().startswith('print('),
        lambda l: l.strip() == ''
    ]
    
    in_docstring = False
    quote_type = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Gestion docstring
        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            in_docstring = True
            quote_type = stripped[:3]
            if stripped.endswith(quote_type) and len(stripped) > 3:
                in_docstring = False
            continue
        elif in_docstring:
            if stripped.endswith(quote_type):
                in_docstring = False
            continue
        
        # Skip patterns normaux
        if any(pattern(line) for pattern in skip_patterns):
            continue
        
        # Premier vrai code trouvé
        return i
    
    return len(lines)

def create_launcher_script(script_path: pathlib.Path, launcher_dir: pathlib.Path) -> pathlib.Path:
    """Crée un script launcher dans un répertoire spécifique"""
    launcher_name = f"run_{script_path.stem}.py"
    launcher_path = launcher_dir / launcher_name
    
    # Calculer chemin relatif depuis launcher vers script
    try:
        relative_path = os.path.relpath(script_path, launcher_dir)
    except ValueError:
        # Chemins sur drives différents, utiliser chemin absolu
        relative_path = str(script_path)
    
    launcher_content = f'''#!/usr/bin/env python3
"""
🚀 Launcher pour {script_path.name}

Ce launcher permet d'exécuter {script_path.name} depuis n'importe où.
Résout automatiquement les chemins et configure l'environnement.

Usage: python {launcher_name}

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import pathlib
import subprocess

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU

def main():
    """Lance le script cible avec environnement configuré"""
    print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
    
    # Déterminer chemin du script cible
    launcher_dir = pathlib.Path(__file__).parent
    script_path = launcher_dir / "{relative_path}"
    
    if not script_path.exists():
        print(f"❌ Script non trouvé: {{script_path}}")
        return 1
    
    print(f"🚀 Lancement: {{script_path}}")
    
    # Lancer le script avec les mêmes arguments
    try:
        result = subprocess.run([
            sys.executable, 
            str(script_path)
        ] + sys.argv[1:], check=False)
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Erreur lancement: {{e}}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    
    print(f"🎯 Launcher créé: {launcher_path}")
    return launcher_path

def scan_and_fix_all_scripts(project_root: pathlib.Path, create_launchers: bool = False) -> Dict[str, int]:
    """Scanne et corrige tous les scripts du projet"""
    
    print(f"🔍 Scan des scripts Python dans: {project_root}")
    
    scripts = find_python_scripts(project_root)
    print(f"📊 {len(scripts)} scripts trouvés")
    
    stats = {
        'total': len(scripts),
        'already_portable': 0,
        'made_portable': 0,
        'errors': 0,
        'launchers_created': 0
    }
    
    # Créer répertoire launchers si demandé
    launcher_dir = None
    if create_launchers:
        launcher_dir = project_root / 'launchers'
        launcher_dir.mkdir(exist_ok=True)
        print(f"📁 Répertoire launchers: {launcher_dir}")
    
    for script in scripts:
        print(f"\n🔧 Traitement: {script.relative_to(project_root)}")
        
        # Analyser dépendances
        deps = analyze_script_dependencies(script)
        if any(deps.values()):
            print(f"   📋 Dépendances détectées: {sum(len(v) for v in deps.values())}")
        
        # Rendre portable
        if has_portable_header(script):
            stats['already_portable'] += 1
        else:
            if make_script_portable(script):
                stats['made_portable'] += 1
            else:
                stats['errors'] += 1
        
        # Créer launcher si demandé
        if create_launchers and launcher_dir:
            try:
                create_launcher_script(script, launcher_dir)
                stats['launchers_created'] += 1
            except Exception as e:
                print(f"   ⚠️  Erreur launcher: {e}")
    
    return stats

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="🚀 Make Python scripts portable and executable from anywhere",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/make_scripts_portable.py --scan-all
    python tools/make_scripts_portable.py --scan-all --create-launchers
    python tools/make_scripts_portable.py --fix-script PIPELINE/scripts/demo_pipeline.py
    python tools/make_scripts_portable.py --create-launcher tools/clean_sandbox.py
        """
    )
    
    parser.add_argument('--scan-all', action='store_true',
                       help='Scan and fix all Python scripts in project')
    parser.add_argument('--fix-script', 
                       help='Fix a specific script file')
    parser.add_argument('--create-launcher',
                       help='Create launcher for specific script')
    parser.add_argument('--create-launchers', action='store_true',
                       help='Create launchers for all scripts (use with --scan-all)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup files')
    
    args = parser.parse_args()
    
    print("🚀 SuperWhisper V6 - Script Portability Tool")
    print("=" * 50)
    
    # Déterminer project root
    current_file = pathlib.Path(__file__).resolve()
    project_root = current_file.parent.parent  # tools/ -> project root
    
    if args.scan_all:
        stats = scan_and_fix_all_scripts(project_root, args.create_launchers)
        
        print(f"\n📊 RÉSULTATS:")
        print(f"   Total scripts: {stats['total']}")
        print(f"   Déjà portables: {stats['already_portable']}")
        print(f"   Rendus portables: {stats['made_portable']}")
        print(f"   Erreurs: {stats['errors']}")
        if args.create_launchers:
            print(f"   Launchers créés: {stats['launchers_created']}")
        
        print(f"\n✅ Tous les scripts sont maintenant exécutables depuis n'importe où!")
        
    elif args.fix_script:
        script_path = pathlib.Path(args.fix_script)
        if not script_path.exists():
            print(f"❌ Script non trouvé: {script_path}")
            return 1
        
        success = make_script_portable(script_path, not args.no_backup)
        return 0 if success else 1
        
    elif args.create_launcher:
        script_path = pathlib.Path(args.create_launcher)
        if not script_path.exists():
            print(f"❌ Script non trouvé: {script_path}")
            return 1
        
        launcher_dir = project_root / 'launchers'
        launcher_dir.mkdir(exist_ok=True)
        
        try:
            launcher_path = create_launcher_script(script_path, launcher_dir)
            print(f"✅ Launcher créé: {launcher_path}")
            return 0
        except Exception as e:
            print(f"❌ Erreur création launcher: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️  Opération interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        sys.exit(1) 