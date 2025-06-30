#!/usr/bin/env python3
"""
🧹 Script de Purge Automatique - Tests Sandbox SuperWhisper V6

Supprime automatiquement les tests exploratoires vieux de plus de 7 jours
dans les répertoires sandbox pour maintenir la propreté du dépôt.

Usage:
    python tools/clean_sandbox.py              # Purge automatique
    python tools/clean_sandbox.py --dry-run    # Simulation sans suppression
    python tools/clean_sandbox.py --days=14    # Seuil personnalisé (14 jours)
    python tools/clean_sandbox.py --force      # Purge tout sans confirmation

Répertoires surveillés:
    - tests/sandbox/
    - PIPELINE/tests/sandbox/
    - .cursor-tmp/

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

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

import time
import pathlib
import shutil
import argparse
from datetime import datetime, timedelta
from typing import List, Tuple

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée pour clean_sandbox.py")

# Configuration par défaut
DEFAULT_THRESHOLD_DAYS = 7
SANDBOX_DIRECTORIES = [
    "tests/sandbox",
    "PIPELINE/tests/sandbox",
    ".cursor-tmp"
]

EXTENSIONS_TO_CLEAN = [
    "*.py", "*.json", "*.txt", "*.log", "*.wav", "*.mp3", 
    "*.tmp", "*.cache", "*.pkl", "*.pt", "*.pth"
]

def get_file_age_days(file_path: pathlib.Path) -> float:
    """Calcule l'âge d'un fichier en jours"""
    try:
        mtime = file_path.stat().st_mtime
        age_seconds = time.time() - mtime
        return age_seconds / 86400  # Convert to days
    except (OSError, FileNotFoundError):
        return 0

def find_old_files(sandbox_dir: pathlib.Path, threshold_days: int) -> List[Tuple[pathlib.Path, float]]:
    """Trouve tous les fichiers anciens dans un répertoire sandbox"""
    old_files = []
    
    if not sandbox_dir.exists():
        return old_files
    
    for pattern in EXTENSIONS_TO_CLEAN:
        for file_path in sandbox_dir.rglob(pattern):
            if file_path.is_file():
                age_days = get_file_age_days(file_path)
                if age_days > threshold_days:
                    old_files.append((file_path, age_days))
    
    return old_files

def clean_pycache(sandbox_dir: pathlib.Path, dry_run: bool = False) -> int:
    """Supprime les répertoires __pycache__ dans sandbox"""
    cleaned = 0
    
    for pycache_dir in sandbox_dir.rglob("__pycache__"):
        if pycache_dir.is_dir():
            if dry_run:
                print(f"🗑️  [DRY-RUN] Would delete __pycache__: {pycache_dir}")
            else:
                try:
                    shutil.rmtree(pycache_dir, ignore_errors=True)
                    print(f"🗑️  Deleted __pycache__: {pycache_dir}")
                    cleaned += 1
                except Exception as e:
                    print(f"❌ Error deleting {pycache_dir}: {e}")
    
    return cleaned

def clean_sandbox_directory(sandbox_dir: pathlib.Path, threshold_days: int, 
                          dry_run: bool = False, force: bool = False) -> Tuple[int, int]:
    """Nettoie un répertoire sandbox spécifique"""
    sandbox_path = pathlib.Path(sandbox_dir)
    
    if not sandbox_path.exists():
        print(f"⚠️  Sandbox directory not found: {sandbox_path}")
        return 0, 0
    
    print(f"\n📂 Scanning: {sandbox_path}")
    
    # Trouver les fichiers anciens
    old_files = find_old_files(sandbox_path, threshold_days)
    
    if not old_files and not any(sandbox_path.rglob("__pycache__")):
        print(f"✅ No files to clean in {sandbox_path}")
        return 0, 0
    
    files_cleaned = 0
    size_freed = 0
    
    # Nettoyer les fichiers anciens
    if old_files:
        print(f"📊 Found {len(old_files)} files older than {threshold_days} days:")
        
        # Afficher preview
        for file_path, age_days in old_files[:5]:  # Limiter à 5 pour preview
            file_size = file_path.stat().st_size if file_path.exists() else 0
            print(f"   📄 {file_path} ({age_days:.1f} days, {file_size:,} bytes)")
        
        if len(old_files) > 5:
            print(f"   ... and {len(old_files) - 5} more files")
        
        # Confirmation si pas en mode force
        if not force and not dry_run:
            response = input(f"\n🤔 Delete {len(old_files)} files? [y/N]: ").lower()
            if response not in ['y', 'yes']:
                print("❌ Aborted by user")
                return 0, 0
        
        # Supprimer les fichiers
        for file_path, age_days in old_files:
            try:
                if dry_run:
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    print(f"🗑️  [DRY-RUN] Would delete: {file_path} ({file_size:,} bytes)")
                    size_freed += file_size
                else:
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    file_path.unlink()
                    print(f"🗑️  Deleted: {file_path} ({file_size:,} bytes)")
                    size_freed += file_size
                    files_cleaned += 1
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
    
    # Nettoyer __pycache__
    pycache_cleaned = clean_pycache(sandbox_path, dry_run)
    
    return files_cleaned, size_freed

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="🧹 Clean old files from sandbox directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/clean_sandbox.py                # Clean files older than 7 days
    python tools/clean_sandbox.py --dry-run      # Show what would be deleted
    python tools/clean_sandbox.py --days=14      # Custom threshold (14 days)
    python tools/clean_sandbox.py --force        # No confirmation prompts
        """
    )
    
    parser.add_argument('--days', type=int, default=DEFAULT_THRESHOLD_DAYS,
                       help=f'Threshold in days (default: {DEFAULT_THRESHOLD_DAYS})')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--force', action='store_true',
                       help='Delete without confirmation prompts')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("🧹 SuperWhisper V6 - Sandbox Cleanup Tool")
    print("=" * 50)
    print(f"📅 Threshold: {args.days} days")
    print(f"🔍 Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"⚡ Force: {'Yes' if args.force else 'No'}")
    
    total_files_cleaned = 0
    total_size_freed = 0
    
    # Nettoyer chaque répertoire sandbox
    for sandbox_dir in SANDBOX_DIRECTORIES:
        files_cleaned, size_freed = clean_sandbox_directory(
            pathlib.Path(sandbox_dir), 
            args.days, 
            args.dry_run, 
            args.force
        )
        total_files_cleaned += files_cleaned
        total_size_freed += size_freed
    
    # Résumé final
    print("\n" + "=" * 50)
    print("📊 CLEANUP SUMMARY")
    print("=" * 50)
    print(f"🗑️  Files cleaned: {total_files_cleaned}")
    print(f"💾 Space freed: {total_size_freed:,} bytes ({total_size_freed/1024/1024:.1f} MB)")
    
    if args.dry_run:
        print("ℹ️  This was a dry-run. No files were actually deleted.")
        print("   Run without --dry-run to perform actual cleanup.")
    
    # Suggestions
    if total_files_cleaned > 0 or args.dry_run:
        print("\n💡 SUGGESTIONS:")
        print("   - Consider moving valuable tests to tests/unit/ or tests/integration/")
        print("   - Run this script weekly via cron job or task scheduler")
        print("   - Add to pre-commit hooks for automatic cleanup")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️  Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 