#!/usr/bin/env python3
"""
Conversion fichiers PCM → WAV - SuperWhisper V6 TTS
Convertit tous les fichiers audio invalides en format WAV standard

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

from pathlib import Path
import shutil

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Import utilitaires audio
from TTS.utils_audio import pcm_to_wav, is_valid_wav, get_wav_info

def convertir_fichiers_pcm_wav():
    """Convertit tous les fichiers PCM invalides en WAV valides"""
    print("\n🔧 CONVERSION FICHIERS PCM → WAV - SUPERWHISPER V6")
    print("=" * 60)
    
    # Répertoire de travail
    test_dir = Path("test_output")
    if not test_dir.exists():
        print("❌ Répertoire test_output introuvable")
        return
    
    # Recherche fichiers WAV
    wav_files = list(test_dir.glob("*.wav"))
    if not wav_files:
        print("❌ Aucun fichier WAV trouvé")
        return
    
    print(f"📁 Analyse de {len(wav_files)} fichiers WAV...")
    
    # Statistiques
    total_files = 0
    invalid_files = 0
    converted_files = 0
    already_valid_files = 0
    error_files = 0
    
    # Création répertoire backup
    backup_dir = test_dir / "backup_pcm_original"
    backup_dir.mkdir(exist_ok=True)
    
    print(f"💾 Backup original → {backup_dir}")
    print("\n" + "=" * 60)
    
    for wav_file in sorted(wav_files):
        total_files += 1
        file_name = wav_file.name
        
        print(f"\n📄 Traitement: {file_name}")
        print("-" * 40)
        
        try:
            # Lecture fichier
            with open(wav_file, 'rb') as f:
                data = f.read()
            
            # Vérification format
            is_valid = is_valid_wav(data)
            
            if is_valid:
                # Fichier déjà valide
                wav_info = get_wav_info(data)
                already_valid_files += 1
                
                print(f"✅ DÉJÀ VALIDE")
                if 'error' not in wav_info:
                    print(f"   Durée: {wav_info.get('duration_ms', 'N/A')}ms")
                    print(f"   Taille: {len(data)} bytes")
                
            else:
                # Fichier invalide - conversion nécessaire
                invalid_files += 1
                print(f"🔧 CONVERSION REQUISE")
                
                # Backup original
                backup_file = backup_dir / file_name
                shutil.copy2(wav_file, backup_file)
                print(f"   Backup: {backup_file.name}")
                
                # Conversion PCM → WAV
                try:
                    # Paramètres par défaut Piper
                    wav_data = pcm_to_wav(
                        pcm_data=data,
                        sample_rate=22050,
                        channels=1,
                        sampwidth=2
                    )
                    
                    # Validation conversion
                    if is_valid_wav(wav_data):
                        # Sauvegarde fichier converti
                        with open(wav_file, 'wb') as f:
                            f.write(wav_data)
                        
                        converted_files += 1
                        wav_info = get_wav_info(wav_data)
                        
                        print(f"   ✅ CONVERTI AVEC SUCCÈS")
                        print(f"   Avant: {len(data)} bytes (PCM)")
                        print(f"   Après: {len(wav_data)} bytes (WAV)")
                        
                        if 'error' not in wav_info:
                            print(f"   Durée: {wav_info.get('duration_ms', 'N/A')}ms")
                            print(f"   Canaux: {wav_info.get('channels', 'N/A')}")
                            print(f"   Fréquence: {wav_info.get('framerate', 'N/A')} Hz")
                    
                    else:
                        print(f"   ❌ ÉCHEC CONVERSION (format toujours invalide)")
                        error_files += 1
                        
                except Exception as e:
                    print(f"   ❌ ERREUR CONVERSION: {e}")
                    error_files += 1
                    
        except Exception as e:
            print(f"❌ ERREUR LECTURE: {e}")
            error_files += 1
    
    # Résumé final
    print(f"\n📊 RÉSUMÉ CONVERSION")
    print("=" * 60)
    print(f"Fichiers analysés:     {total_files}")
    print(f"Déjà valides:          {already_valid_files}")
    print(f"Invalides détectés:    {invalid_files}")
    print(f"Conversions réussies:  {converted_files}")
    print(f"Erreurs:               {error_files}")
    print("-" * 60)
    
    if converted_files > 0:
        print(f"🎉 SUCCÈS: {converted_files} fichiers convertis avec succès!")
        print(f"💾 Originaux sauvés dans: {backup_dir}")
        print(f"🎵 Tous les fichiers sont maintenant audibles!")
    elif already_valid_files == total_files:
        print(f"✅ PARFAIT: Tous les fichiers étaient déjà valides!")
    else:
        print(f"⚠️  ATTENTION: {error_files} fichiers n'ont pas pu être convertis")
    
    return converted_files

def tester_fichiers_convertis():
    """Test rapide des fichiers convertis"""
    print(f"\n🧪 TEST FICHIERS CONVERTIS")
    print("=" * 60)
    
    test_dir = Path("test_output")
    wav_files = list(test_dir.glob("*.wav"))
    
    if not wav_files:
        print("❌ Aucun fichier à tester")
        return
    
    valid_count = 0
    invalid_count = 0
    
    for wav_file in sorted(wav_files):
        try:
            with open(wav_file, 'rb') as f:
                data = f.read()
            
            is_valid = is_valid_wav(data)
            
            if is_valid:
                valid_count += 1
                wav_info = get_wav_info(data)
                duration = wav_info.get('duration_ms', 0) if 'error' not in wav_info else 0
                size_mb = len(data) / 1024 / 1024
                
                print(f"✅ {wav_file.name:30} | {duration:5.0f}ms | {size_mb:6.2f}MB")
            else:
                invalid_count += 1
                print(f"❌ {wav_file.name:30} | INVALIDE")
                
        except Exception as e:
            invalid_count += 1
            print(f"❌ {wav_file.name:30} | ERREUR: {e}")
    
    print("-" * 60)
    print(f"Fichiers valides: {valid_count}")
    print(f"Fichiers invalides: {invalid_count}")
    
    if invalid_count == 0:
        print("🎉 PARFAIT: Tous les fichiers sont maintenant valides et audibles!")
    else:
        print(f"⚠️  {invalid_count} fichiers restent problématiques")

def convertir_fichier_principal():
    """Conversion prioritaire du fichier principal validation_utilisateur_complet.wav"""
    print(f"\n🎯 CONVERSION PRIORITAIRE FICHIER PRINCIPAL")
    print("=" * 60)
    
    fichier_principal = Path("test_output/validation_utilisateur_complet.wav")
    
    if not fichier_principal.exists():
        print("❌ Fichier principal introuvable")
        return False
    
    print(f"📄 Fichier: {fichier_principal.name}")
    
    try:
        # Lecture fichier
        with open(fichier_principal, 'rb') as f:
            data = f.read()
        
        print(f"📊 Taille originale: {len(data)} bytes ({len(data)/1024/1024:.2f} MB)")
        
        # Vérification format
        if is_valid_wav(data):
            print("✅ Fichier déjà au format WAV valide")
            wav_info = get_wav_info(data)
            if 'error' not in wav_info:
                print(f"   Durée: {wav_info.get('duration_ms', 'N/A')}ms")
                print(f"   Qualité: {wav_info.get('framerate', 'N/A')} Hz, {wav_info.get('channels', 'N/A')} canaux")
            return True
        
        print("🔧 Conversion PCM → WAV en cours...")
        
        # Backup original
        backup_file = fichier_principal.with_suffix('.pcm.backup')
        shutil.copy2(fichier_principal, backup_file)
        print(f"💾 Backup: {backup_file.name}")
        
        # Conversion avec paramètres optimaux
        wav_data = pcm_to_wav(
            pcm_data=data,
            sample_rate=22050,  # Standard Piper
            channels=1,         # Mono
            sampwidth=2         # 16-bit
        )
        
        # Validation
        if is_valid_wav(wav_data):
            # Sauvegarde
            with open(fichier_principal, 'wb') as f:
                f.write(wav_data)
            
            wav_info = get_wav_info(wav_data)
            
            print("✅ CONVERSION RÉUSSIE!")
            print(f"   Avant: {len(data)} bytes (PCM brut)")
            print(f"   Après: {len(wav_data)} bytes (WAV standard)")
            
            if 'error' not in wav_info:
                print(f"   Durée: {wav_info.get('duration_ms', 'N/A')}ms")
                print(f"   Qualité: {wav_info.get('framerate', 'N/A')} Hz, {wav_info.get('channels', 'N/A')} canaux")
            
            print(f"🎵 Le fichier est maintenant audible!")
            return True
        
        else:
            print("❌ Échec conversion - format toujours invalide")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("🚀 DÉMARRAGE CONVERSION FICHIERS PCM → WAV")
    
    # 1. Conversion prioritaire fichier principal
    print("\n" + "🎯" * 20 + " PHASE 1: FICHIER PRINCIPAL " + "🎯" * 20)
    success_principal = convertir_fichier_principal()
    
    # 2. Conversion tous les fichiers
    print("\n" + "🔧" * 20 + " PHASE 2: TOUS LES FICHIERS " + "🔧" * 20)
    converted_count = convertir_fichiers_pcm_wav()
    
    # 3. Test final
    print("\n" + "🧪" * 20 + " PHASE 3: VALIDATION FINALE " + "🧪" * 20)
    tester_fichiers_convertis()
    
    # Résumé global
    print(f"\n🏁 CONVERSION TERMINÉE")
    print("=" * 60)
    
    if success_principal:
        print("✅ Fichier principal: CONVERTI ET AUDIBLE")
    else:
        print("❌ Fichier principal: PROBLÈME PERSISTANT")
    
    if converted_count > 0:
        print(f"✅ Fichiers additionnels: {converted_count} CONVERTIS")
    
    print("\n🎵 Vous pouvez maintenant écouter tous les fichiers WAV!")
    print("💾 Les originaux sont sauvés dans test_output/backup_pcm_original/") 