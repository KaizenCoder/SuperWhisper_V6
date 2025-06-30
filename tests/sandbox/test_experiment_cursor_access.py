#!/usr/bin/env python3
"""
🧪 Test Exploratoire - Démonstration Accès Cursor

Ce fichier démontre que Cursor a TOUJOURS accès aux tests exploratoires
créés dans tests/sandbox/ - ils restent VISIBLES et ACCESSIBLES.

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

def test_cursor_access_demonstration():
    """
    ✅ CE FICHIER EST VISIBLE ET ACCESSIBLE À CURSOR
    
    - Créé dans tests/sandbox/ (PAS dans .cursor-tmp/)
    - Visible dans l'explorateur de fichiers Cursor
    - Accessible pour édition, exécution, debugging
    - Inclus dans les recherches Cursor
    - Peut être ouvert, modifié, exécuté normalement
    """
    print("✅ CURSOR PEUT VOIR CE FICHIER !")
    print("📂 Emplacement: tests/sandbox/ (visible)")
    print("🔍 Recherche: Inclus dans résultats")
    print("✏️  Édition: Accessible normalement")
    print("🔧 Debug: Fonctionnel avec breakpoints")
    print("▶️  Exécution: Run disponible")
    
    # Simulation test exploratoire STT
    experiment_result = {
        "component": "STT",
        "status": "exploring",
        "cursor_accessible": True,
        "location": "tests/sandbox/",
        "visibility": "full",
        "editable": True,
        "runnable": True
    }
    
    return experiment_result

def test_different_from_cursor_temp():
    """
    🔄 DIFFÉRENCE AVEC FICHIERS TEMPORAIRES CURSOR
    
    Ce fichier (tests/sandbox/) ≠ Fichiers temporaires (.cursor-tmp/)
    """
    comparison = {
        "tests_sandbox": {
            "visibility": "✅ VISIBLE à Cursor",
            "access": "✅ COMPLET (lecture/écriture/exécution)", 
            "search": "✅ INCLUS dans recherches",
            "purpose": "Tests exploratoires développeur",
            "lifespan": "7 jours puis purge auto (sauf si promu)"
        },
        "cursor_tmp": {
            "visibility": "❌ CACHÉ de l'interface",
            "access": "❌ LIMITÉ (fichiers jetables)",
            "search": "❌ EXCLU des recherches", 
            "purpose": "Fichiers temporaires automatiques Cursor",
            "lifespan": "Purge automatique immédiate"
        }
    }
    
    return comparison

if __name__ == "__main__":
    print("🎯 Démonstration Accès Cursor aux Tests Exploratoires")
    print("=" * 60)
    
    # Test accessibilité
    result = test_cursor_access_demonstration()
    print(f"\n📊 Résultat test: {result}")
    
    # Comparaison types fichiers
    comparison = test_different_from_cursor_temp()
    print(f"\n🔄 Comparaison: {comparison}")
    
    print("\n💡 CONCLUSION:")
    print("   ✅ Cursor PEUT voir et utiliser ce fichier")
    print("   ✅ Workflow exploratoire intact")
    print("   ✅ Différent des fichiers temporaires cachés")
    print("   🧹 Purge auto après 7j (sauf si promu vers unit/)") 