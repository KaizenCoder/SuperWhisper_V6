#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploration de l'API Piper pour découvrir la bonne utilisation
"""

import sys
import importlib
import pkgutil

def explore_piper():
    print("🔍 Exploration de l'API Piper")
    print("=" * 40)
    
    try:
        import piper
        print(f"✅ Module piper importé: {piper}")
        print(f"   Chemin: {piper.__file__}")
        print(f"   Package path: {getattr(piper, '__path__', 'N/A')}")
        
        # Explorer les sous-modules
        print("\n📦 Sous-modules disponibles:")
        if hasattr(piper, '__path__'):
            for importer, name, ispkg in pkgutil.iter_modules(piper.__path__, piper.__name__ + '.'):
                print(f"   - {name} (package: {ispkg})")
                
                # Essayer d'importer chaque sous-module
                try:
                    module = importlib.import_module(name)
                    attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                    print(f"     Attributs: {attrs[:5]}{'...' if len(attrs) > 5 else ''}")
                except Exception as e:
                    print(f"     Erreur import: {e}")
        
        # Essayer des imports possibles
        print("\n🧪 Test d'imports possibles:")
        
        possible_imports = [
            'piper.voice',
            'piper.PiperVoice', 
            'piper.Voice',
            'piper.tts',
            'piper.synthesize'
        ]
        
        for imp in possible_imports:
            try:
                module = importlib.import_module(imp.split('.')[0])
                if '.' in imp:
                    parts = imp.split('.')
                    obj = module
                    for part in parts[1:]:
                        obj = getattr(obj, part)
                print(f"   ✅ {imp}: {type(obj)}")
            except (ImportError, AttributeError) as e:
                print(f"   ❌ {imp}: {e}")
        
        # Examiner les fichiers dans le répertoire piper du projet
        print("\n📁 Exploration du répertoire piper local:")
        import os
        piper_dir = "./piper/src/python"
        if os.path.exists(piper_dir):
            for root, dirs, files in os.walk(piper_dir):
                for file in files:
                    if file.endswith('.py'):
                        rel_path = os.path.relpath(os.path.join(root, file), piper_dir)
                        print(f"   - {rel_path}")
        else:
            print("   Répertoire piper/src/python non trouvé")
            
            # Explorer le contenu du dossier piper/src
            if os.path.exists("./piper/src"):
                print("   Contenu de piper/src:")
                for item in os.listdir("./piper/src"):
                    print(f"     - {item}")
                    
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exploration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_piper_cli():
    """Test de l'utilisation de Piper en ligne de commande"""
    print("\n🔧 Test de Piper CLI:")
    
    import subprocess
    import os
    
    # Chercher l'exécutable piper
    piper_paths = [
        "./piper/build/piper",
        "./piper/build/Release/piper.exe",
        "./piper/build/Debug/piper.exe"
    ]
    
    for path in piper_paths:
        if os.path.exists(path):
            print(f"   ✅ Exécutable trouvé: {path}")
            try:
                result = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=5)
                print(f"   Aide: {result.stdout[:200]}...")
                return path
            except Exception as e:
                print(f"   Erreur exécution: {e}")
        else:
            print(f"   ❌ Non trouvé: {path}")
    
    return None

if __name__ == "__main__":
    explore_piper()
    test_piper_cli() 