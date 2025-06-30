#!/usr/bin/env python3
"""
🔄 Auto-Fix Nouveaux Scripts - SuperWhisper V6

Surveille et rend automatiquement portables les nouveaux scripts Python
créés dans le projet, avec intégration Git hooks et surveillance continue.

Usage:
    python tools/auto_fix_new_scripts.py --watch          # Surveillance continue
    python tools/auto_fix_new_scripts.py --git-hook       # Installation Git hook
    python tools/auto_fix_new_scripts.py --fix-recent     # Fix scripts récents (24h)

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import time
import pathlib
import subprocess
import argparse
import json
from datetime import datetime, timedelta
from typing import Set, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée pour auto_fix_new_scripts.py")

# Déterminer racine projet
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)

class PythonFileHandler(FileSystemEventHandler):
    """Handler pour surveiller création/modification fichiers Python"""
    
    def __init__(self):
        self.processed_files: Set[str] = set()
        self.make_portable_script = PROJECT_ROOT / "tools" / "make_scripts_portable.py"
        
    def on_created(self, event):
        """Nouveau fichier créé"""
        if not event.is_directory and event.src_path.endswith('.py'):
            self._process_file(event.src_path, "CRÉÉ")
            
    def on_modified(self, event):
        """Fichier modifié"""
        if not event.is_directory and event.src_path.endswith('.py'):
            # Éviter traitement multiple du même fichier
            if event.src_path not in self.processed_files:
                self._process_file(event.src_path, "MODIFIÉ")
                
    def _process_file(self, file_path: str, action: str):
        """Traiter un fichier Python"""
        try:
            file_path_obj = pathlib.Path(file_path)
            
            # Ignorer fichiers temporaires et cache
            if any(part.startswith('.') for part in file_path_obj.parts):
                return
            if '__pycache__' in str(file_path_obj):
                return
            if file_path_obj.name.startswith('tmp_') or file_path_obj.name.startswith('scratch_'):
                return
                
            print(f"\n🔄 {action}: {file_path_obj.relative_to(PROJECT_ROOT)}")
            
            # Rendre portable avec make_scripts_portable.py
            result = subprocess.run([
                sys.executable, str(self.make_portable_script),
                "--fix-script", str(file_path_obj)
            ], capture_output=True, text=True, cwd=PROJECT_ROOT)
            
            if result.returncode == 0:
                print(f"✅ Script rendu portable: {file_path_obj.name}")
                self.processed_files.add(file_path)
            else:
                print(f"⚠️ Erreur portabilité: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Erreur traitement {file_path}: {e}")

def watch_directory():
    """Surveillance continue des nouveaux fichiers Python"""
    print(f"👁️ Surveillance continue démarrée sur: {PROJECT_ROOT}")
    print("📝 Nouveaux scripts Python seront automatiquement rendus portables")
    print("🛑 Ctrl+C pour arrêter\n")
    
    event_handler = PythonFileHandler()
    observer = Observer()
    
    # Surveiller récursivement tout le projet
    observer.schedule(event_handler, str(PROJECT_ROOT), recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Surveillance arrêtée")
        observer.stop()
    observer.join()

def install_git_hook():
    """Installer Git hook pour auto-fix des nouveaux scripts"""
    git_hooks_dir = PROJECT_ROOT / ".git" / "hooks"
    pre_commit_hook = git_hooks_dir / "pre-commit"
    
    if not git_hooks_dir.exists():
        print("❌ Répertoire .git/hooks non trouvé")
        return False
        
    hook_content = f'''#!/bin/bash
# Auto-fix nouveaux scripts Python - SuperWhisper V6
echo "🔄 Vérification portabilité scripts Python..."

# Obtenir fichiers Python modifiés
python_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\\.py$')

if [ -n "$python_files" ]; then
    echo "📝 Scripts Python détectés, vérification portabilité..."
    cd "{PROJECT_ROOT}"
    python tools/auto_fix_new_scripts.py --fix-staged
    
    # Re-stage les fichiers modifiés
    for file in $python_files; do
        if [ -f "$file" ]; then
            git add "$file"
        fi
    done
    
    echo "✅ Scripts rendus portables et re-stagés"
fi
'''
    
    try:
        with open(pre_commit_hook, 'w', encoding='utf-8') as f:
            f.write(hook_content)
        
        # Rendre exécutable (Windows)
        if os.name == 'nt':
            os.chmod(pre_commit_hook, 0o755)
        else:
            os.chmod(pre_commit_hook, 0o755)
            
        print(f"✅ Git hook installé: {pre_commit_hook}")
        print("🎯 Les nouveaux scripts seront automatiquement rendus portables lors des commits")
        return True
        
    except Exception as e:
        print(f"❌ Erreur installation Git hook: {e}")
        return False

def fix_recent_scripts(hours: int = 24):
    """Fixer les scripts créés/modifiés récemment"""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    recent_scripts = []
    
    print(f"🔍 Recherche scripts modifiés dans les dernières {hours}h...")
    
    for py_file in PROJECT_ROOT.rglob("*.py"):
        try:
            # Ignorer fichiers temporaires et cache
            if any(part.startswith('.') for part in py_file.parts):
                continue
            if '__pycache__' in str(py_file):
                continue
                
            # Vérifier date modification
            mtime = datetime.fromtimestamp(py_file.stat().st_mtime)
            if mtime > cutoff_time:
                recent_scripts.append(py_file)
                
        except Exception:
            continue
    
    if not recent_scripts:
        print(f"✅ Aucun script récent trouvé (dernières {hours}h)")
        return
        
    print(f"📝 {len(recent_scripts)} scripts récents trouvés")
    
    # Traiter avec make_scripts_portable.py
    make_portable_script = PROJECT_ROOT / "tools" / "make_scripts_portable.py"
    
    for script in recent_scripts:
        try:
            result = subprocess.run([
                sys.executable, str(make_portable_script),
                "--fix-script", str(script)
            ], capture_output=True, text=True, cwd=PROJECT_ROOT)
            
            if result.returncode == 0:
                print(f"✅ {script.relative_to(PROJECT_ROOT)}")
            else:
                print(f"⚠️ {script.relative_to(PROJECT_ROOT)}: {result.stderr}")
                
        except Exception as e:
            print(f"❌ {script.relative_to(PROJECT_ROOT)}: {e}")

def fix_staged_files():
    """Fixer les fichiers Python stagés pour commit"""
    try:
        # Obtenir fichiers Python stagés
        result = subprocess.run([
            "git", "diff", "--cached", "--name-only", "--diff-filter=ACM"
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        if result.returncode != 0:
            print("❌ Erreur Git diff")
            return
            
        staged_files = [f for f in result.stdout.strip().split('\n') if f.endswith('.py')]
        
        if not staged_files:
            print("✅ Aucun fichier Python stagé")
            return
            
        print(f"📝 {len(staged_files)} fichiers Python stagés")
        
        make_portable_script = PROJECT_ROOT / "tools" / "make_scripts_portable.py"
        
        for file_path in staged_files:
            full_path = PROJECT_ROOT / file_path
            if full_path.exists():
                subprocess.run([
                    sys.executable, str(make_portable_script),
                    "--fix-script", str(full_path)
                ], cwd=PROJECT_ROOT)
                print(f"✅ {file_path}")
                
    except Exception as e:
        print(f"❌ Erreur traitement fichiers stagés: {e}")

def main():
    parser = argparse.ArgumentParser(description="Auto-fix nouveaux scripts Python")
    parser.add_argument("--watch", action="store_true", help="Surveillance continue")
    parser.add_argument("--git-hook", action="store_true", help="Installer Git hook")
    parser.add_argument("--fix-recent", type=int, default=24, help="Fixer scripts récents (heures)")
    parser.add_argument("--fix-staged", action="store_true", help="Fixer fichiers stagés Git")
    
    args = parser.parse_args()
    
    print("🔄 Auto-Fix Nouveaux Scripts - SuperWhisper V6")
    print("=" * 50)
    
    if args.watch:
        watch_directory()
    elif args.git_hook:
        install_git_hook()
    elif args.fix_staged:
        fix_staged_files()
    elif args.fix_recent:
        fix_recent_scripts(args.fix_recent)
    else:
        print("Usage:")
        print("  --watch          Surveillance continue")
        print("  --git-hook       Installer Git hook")
        print("  --fix-recent N   Fixer scripts récents (N heures)")
        print("  --fix-staged     Fixer fichiers stagés Git")

if __name__ == "__main__":
    main() 