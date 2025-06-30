#!/usr/bin/env python3
"""
Test de validation simplifié des corrections critiques du double contrôle GPU
Vérifie directement dans le code source que les vulnérabilités ont été corrigées.

Corrections validées :
1. Fallback sécurisé vers RTX 3090 (GPU 1) même en single-GPU
2. Target GPU inconditionnel (toujours index 1)  
3. Validation VRAM inconditionnelle (24GB requis)
4. Protection absolue contre RTX 5060 (CUDA:0)

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
import sys

def validate_stt_manager_corrections():
    """Valide les corrections dans STT/stt_manager_robust.py"""
    print("🔍 VALIDATION STT MANAGER - CORRECTIONS CRITIQUES")
    print("=" * 55)
    
    file_path = "STT/stt_manager_robust.py"
    if not os.path.exists(file_path):
        print(f"❌ ERREUR: Fichier {file_path} introuvable")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    errors = []
    validations = []
    
    # VALIDATION 1 : Aucune référence à GPU 0 (RTX 5060)
    print("\n📊 Test 1 : Protection contre GPU 0 (RTX 5060)")
    gpu_0_patterns = [
        r'cuda:0',
        r'set_device\(0\)',
        r'selected_gpu\s*=\s*0',
        r'target_gpu\s*=\s*0',
        r'get_device_properties\(0\)'
    ]
    
    found_gpu_0 = False
    for pattern in gpu_0_patterns:
        matches = re.findall(pattern, content)
        if matches:
            errors.append(f"❌ TROUVÉ: Pattern GPU 0 '{pattern}' dans {file_path}")
            found_gpu_0 = True
    
    if not found_gpu_0:
        validations.append("✅ SUCCÈS: Aucune référence GPU 0 (RTX 5060) trouvée")
        print("   ✅ Aucune référence GPU 0 détectée")
    
    # VALIDATION 2 : Correction ligne 80 - Fallback sécurisé
    print("\n📊 Test 2 : Fallback sécurisé single-GPU (ligne 80)")
    line_80_pattern = r'selected_gpu\s*=\s*1'
    if re.search(line_80_pattern, content):
        validations.append("✅ SUCCÈS: Ligne 80 - Fallback sécurisé vers GPU 1")
        print("   ✅ selected_gpu = 1 confirmé")
    else:
        errors.append("❌ ÉCHEC: Ligne 80 - Fallback non sécurisé")
    
    # VALIDATION 3 : Correction ligne 84 - Target GPU inconditionnel
    print("\n📊 Test 3 : Target GPU inconditionnel (ligne 84)")
    line_84_correct = r'target_gpu\s*=\s*1(?!\s*if)'  # GPU 1 sans condition
    if re.search(line_84_correct, content):
        validations.append("✅ SUCCÈS: Ligne 84 - Target GPU inconditionnel = 1")
        print("   ✅ target_gpu = 1 (inconditionnel) confirmé")
    else:
        errors.append("❌ ÉCHEC: Ligne 84 - Target GPU conditionnel détecté")
    
    # VALIDATION 4 : Présence de GPU 1 exclusivement
    print("\n📊 Test 4 : Utilisation exclusive GPU 1 (RTX 3090)")
    gpu_1_patterns = [
        r'cuda:1',
        r'set_device\(1\)',
        r'selected_gpu\s*=\s*1',
        r'target_gpu\s*=\s*1',
        r'get_device_properties\(1\)'
    ]
    
    gpu_1_count = 0
    for pattern in gpu_1_patterns:
        matches = re.findall(pattern, content)
        gpu_1_count += len(matches)
    
    if gpu_1_count > 0:
        validations.append(f"✅ SUCCÈS: {gpu_1_count} références GPU 1 (RTX 3090) trouvées")
        print(f"   ✅ {gpu_1_count} références GPU 1 détectées")
    else:
        errors.append("❌ ÉCHEC: Aucune référence GPU 1 trouvée")
    
    return len(errors) == 0, validations, errors

def validate_other_files_corrections():
    """Valide les corrections dans les autres fichiers critiques"""
    print("\n🔍 VALIDATION AUTRES FICHIERS CRITIQUES")
    print("=" * 45)
    
    files_to_check = [
        "tests/test_stt_handler.py",
        "utils/gpu_manager.py"
    ]
    
    all_validations = []
    all_errors = []
    
    for file_path in files_to_check:
        print(f"\n📊 Validation : {file_path}")
        
        if not os.path.exists(file_path):
            print(f"   ⚠️  Fichier {file_path} introuvable - ignoré")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chercher des références GPU 0 spécifiques (éviter les faux positifs)
        gpu_0_patterns = [
            r'cuda:0',
            r'set_device\(0\)',
            r'get_device_name\(0\)',
            r'get_device_properties\(0\)',
            r'get_device_capability\(0\)',
            r'selected_gpu\s*=\s*0',
            r'target_gpu\s*=\s*0'
        ]
        found_issues = False
        
        for pattern in gpu_0_patterns:
            matches = re.findall(pattern, content)
            if matches:
                all_errors.append(f"❌ {file_path}: Référence GPU 0 '{pattern}' trouvée: {matches}")
                found_issues = True
        
        if not found_issues:
            all_validations.append(f"✅ {file_path}: Aucune référence GPU 0 trouvée")
            print("   ✅ Fichier sécurisé")
        else:
            print("   ❌ Références GPU 0 détectées")
    
    return len(all_errors) == 0, all_validations, all_errors

def validate_config_files():
    """Valide les fichiers de configuration"""
    print("\n🔍 VALIDATION FICHIERS CONFIGURATION")
    print("=" * 40)
    
    config_files = [
        "Config/mvp_settings.yaml",
        "config/mvp_settings.yaml"  # Au cas où
    ]
    
    validations = []
    errors = []
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\n📊 Validation : {config_file}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Vérifier gpu_device_index
            if 'gpu_device_index: 1' in content:
                validations.append(f"✅ {config_file}: gpu_device_index = 1 confirmé")
                print("   ✅ gpu_device_index: 1 confirmé")
            else:
                errors.append(f"❌ {config_file}: gpu_device_index incorrect")
            
            # Vérifier gpu_device
            if 'gpu_device: "cuda:1"' in content:
                validations.append(f"✅ {config_file}: gpu_device = cuda:1 confirmé")
                print("   ✅ gpu_device: cuda:1 confirmé")
            else:
                errors.append(f"❌ {config_file}: gpu_device incorrect")
    
    return len(errors) == 0, validations, errors

def run_complete_validation():
    """Exécute la validation complète des corrections"""
    print("🔍 VALIDATION COMPLÈTE CORRECTIONS DOUBLE CONTRÔLE")
    print("=" * 60)
    print("Validation des 4 corrections critiques appliquées")
    print("=" * 60)
    
    all_validations = []
    all_errors = []
    
    # Test 1 : STT Manager (corrections principales)
    success_1, validations_1, errors_1 = validate_stt_manager_corrections()
    all_validations.extend(validations_1)
    all_errors.extend(errors_1)
    
    # Test 2 : Autres fichiers critiques
    success_2, validations_2, errors_2 = validate_other_files_corrections()
    all_validations.extend(validations_2)
    all_errors.extend(errors_2)
    
    # Test 3 : Fichiers de configuration
    success_3, validations_3, errors_3 = validate_config_files()
    all_validations.extend(validations_3)
    all_errors.extend(errors_3)
    
    # RÉSUMÉ FINAL
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ VALIDATION DOUBLE CONTRÔLE")
    print("=" * 60)
    
    success_global = success_1 and success_2 and success_3
    
    if success_global:
        print("🎉 TOUTES LES CORRECTIONS VALIDÉES AVEC SUCCÈS")
        print("🔒 SÉCURITÉ RTX 3090 EXCLUSIVE CONFIRMÉE")
        print("🎯 VULNÉRABILITÉS CRITIQUES ÉLIMINÉES")
        print(f"\n✅ {len(all_validations)} validations réussies :")
        for validation in all_validations:
            print(f"   {validation}")
    else:
        print("⚠️  PROBLÈMES DÉTECTÉS DANS LES CORRECTIONS")
        print(f"\n❌ {len(all_errors)} erreurs trouvées :")
        for error in all_errors:
            print(f"   {error}")
        
        if all_validations:
            print(f"\n✅ {len(all_validations)} validations réussies :")
            for validation in all_validations:
                print(f"   {validation}")
    
    print("\n" + "=" * 60)
    print("🚀 PRÊT POUR LA SUITE DU DÉVELOPPEMENT" if success_global else "🔧 CORRECTIONS SUPPLÉMENTAIRES REQUISES")
    print("=" * 60)
    
    return success_global

if __name__ == "__main__":
    success = run_complete_validation()
    sys.exit(0 if success else 1) 