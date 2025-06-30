#!/usr/bin/env python3
"""
Démonstration TTS - SuperWhisper V6
Script de test manuel avec génération de fichier audio pour écoute réelle
🎵 Validation qualité audio en conditions réelles
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Import du système TTS
try:
    # Ajout du chemin du projet au sys.path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from TTS.tts_manager import UnifiedTTSManager
    from TTS.utils_audio import is_valid_wav, get_wav_info
    import yaml
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Système TTS non disponible: {e}")
    TTS_AVAILABLE = False

async def demo_tts_interactive():
    """Démonstration interactive du TTS avec génération de fichier audio"""
    print("\n" + "="*80)
    print("🎵 DÉMONSTRATION TTS SUPERWHISPER V6")
    print("="*80)
    print("🚀 Test manuel avec génération de fichier audio pour écoute réelle")
    print()
    
    if not TTS_AVAILABLE:
        print("❌ Système TTS non disponible")
        return
    
    try:
        # Initialisation du TTS Manager
        print("🔧 Initialisation du TTS Manager...")
        config_path = Path("config/tts.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        start_time = time.perf_counter()
        tts_manager = UnifiedTTSManager(config)
        init_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ TTS Manager initialisé en {init_time:.1f}ms")
        print()
        
        # Textes de démonstration prédéfinis
        demo_texts = {
            '1': "Bonjour ! Bienvenue dans la démonstration de SuperWhisper V6.",
            '2': "Ce système utilise des technologies avancées d'intelligence artificielle pour offrir une synthèse vocale de haute qualité avec des performances optimisées.",
            '3': """SuperWhisper V6 Phase 3 intègre plusieurs optimisations majeures : 
                 le binding Python natif pour réduire la latence, 
                 un cache LRU intelligent pour les textes récurrents, 
                 le chunking sémantique pour les textes longs, 
                 et une optimisation GPU dédiée pour maximiser les performances.""",
            '4': """L'intelligence artificielle conversationnelle représente l'avenir des interactions homme-machine. 
                 SuperWhisper V6 s'inscrit dans cette dynamique en proposant une solution complète et optimisée 
                 pour les interactions vocales. Le système repose sur une architecture en pipeline comprenant 
                 trois composants principaux : reconnaissance vocale, traitement du langage naturel et synthèse vocale 
                 multi-backend avec fallback intelligent. Les optimisations Phase 3 permettent d'atteindre 
                 des performances exceptionnelles tout en maintenant une qualité audio remarquable.""",
            '5': "Test de caractères spéciaux : café, naïve, 123 euros, @#%&*()[]{}|\\:;\"'<>,.?/~`",
        }
        
        while True:
            print("🎵 MENU DÉMONSTRATION TTS")
            print("-" * 40)
            print("Choisissez une option :")
            print()
            print("📝 TEXTES PRÉDÉFINIS:")
            for key, text in demo_texts.items():
                preview = text[:60] + "..." if len(text) > 60 else text
                print(f"   {key}. {preview}")
            print()
            print("⌨️  SAISIE PERSONNALISÉE:")
            print("   c. Saisir un texte personnalisé")
            print("   q. Quitter")
            print()
            
            choice = input("Votre choix : ").strip().lower()
            
            if choice == 'q':
                print("👋 Au revoir !")
                break
            elif choice == 'c':
                print("\n📝 Saisie personnalisée:")
                print("Entrez votre texte (ou 'annuler' pour revenir au menu) :")
                custom_text = input("> ").strip()
                if custom_text.lower() == 'annuler' or not custom_text:
                    continue
                text_to_synthesize = custom_text
                demo_name = "personnalise"
            elif choice in demo_texts:
                text_to_synthesize = demo_texts[choice]
                demo_name = f"demo_{choice}"
            else:
                print("❌ Choix invalide. Veuillez réessayer.")
                continue
            
            # Synthèse TTS
            print(f"\n🔄 Synthèse en cours...")
            print(f"📝 Texte ({len(text_to_synthesize)} caractères):")
            print(f"   {text_to_synthesize[:100]}{'...' if len(text_to_synthesize) > 100 else ''}")
            print()
            
            try:
                # Mesure de performance
                start_synthesis = time.perf_counter()
                tts_result = await tts_manager.synthesize(text_to_synthesize)
                synthesis_time = (time.perf_counter() - start_synthesis) * 1000
                
                # Extraction des données audio
                if hasattr(tts_result, 'audio_data'):
                    wav_bytes = tts_result.audio_data
                else:
                    wav_bytes = tts_result
                
                # Validation du format
                is_valid = is_valid_wav(wav_bytes)
                audio_info = get_wav_info(wav_bytes)
                
                # Génération du fichier de sortie
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"demo_tts_{demo_name}_{timestamp}.wav"
                output_path = Path(output_filename)
                
                with open(output_path, "wb") as f:
                    f.write(wav_bytes)
                
                # Affichage des résultats
                print("✅ SYNTHÈSE RÉUSSIE !")
                print("-" * 40)
                print(f"⚡ Temps de synthèse: {synthesis_time:.1f}ms")
                print(f"📊 Taille audio: {len(wav_bytes):,} bytes")
                print(f"🎵 Durée audio: {audio_info.get('duration_ms', 0):.0f}ms")
                print(f"✅ Format WAV valide: {is_valid}")
                print(f"💾 Fichier généré: {output_path}")
                print()
                
                # Calcul de métriques
                chars_per_second = len(text_to_synthesize) / (synthesis_time / 1000)
                print(f"📈 MÉTRIQUES:")
                print(f"   Débit: {chars_per_second:.1f} caractères/seconde")
                
                # Estimation du backend utilisé
                if synthesis_time < 10:
                    backend_used = "Cache (hit)"
                elif synthesis_time < 500:
                    backend_used = "Piper Native"
                elif synthesis_time < 1500:
                    backend_used = "Piper CLI"
                else:
                    backend_used = "SAPI/Fallback"
                
                print(f"   Backend estimé: {backend_used}")
                print()
                
                # Instructions d'écoute
                print("🎧 ÉCOUTE DU RÉSULTAT:")
                print(f"   1. Ouvrez le fichier: {output_path}")
                print("   2. Écoutez la qualité audio")
                print("   3. Vérifiez la prononciation et la fluidité")
                print("   4. Validez que le texte est correctement synthétisé")
                print()
                
                # Analyse automatique de la qualité
                if audio_info.get('duration_ms', 0) > 0:
                    expected_duration = len(text_to_synthesize) * 60  # ~60ms par caractère
                    duration_ratio = audio_info['duration_ms'] / expected_duration
                    
                    if 0.5 <= duration_ratio <= 2.0:
                        quality_status = "✅ Durée cohérente"
                    else:
                        quality_status = f"⚠️ Durée suspecte (ratio: {duration_ratio:.1f})"
                    
                    print(f"🔍 ANALYSE AUTOMATIQUE:")
                    print(f"   {quality_status}")
                    print()
                
            except Exception as e:
                print(f"❌ Erreur lors de la synthèse: {e}")
                print()
            
            # Pause avant le prochain test
            input("Appuyez sur Entrée pour continuer...")
            print("\n" + "="*80)
        
        # Nettoyage
        await tts_manager.cleanup()
        
    except Exception as e:
        print(f"❌ Erreur démonstration: {e}")
        import traceback
        traceback.print_exc()

async def demo_tts_batch():
    """Démonstration batch avec génération de plusieurs fichiers de test"""
    print("\n" + "="*80)
    print("🎵 DÉMONSTRATION TTS BATCH - GÉNÉRATION MULTIPLE")
    print("="*80)
    
    if not TTS_AVAILABLE:
        print("❌ Système TTS non disponible")
        return
    
    # Textes de test variés
    test_cases = [
        ("court", "Bonjour, test de synthèse vocale."),
        ("moyen", "SuperWhisper V6 utilise l'intelligence artificielle pour offrir une synthèse vocale de haute qualité."),
        ("long", """L'intelligence artificielle conversationnelle représente une révolution dans les interactions homme-machine. 
                   SuperWhisper V6 intègre des technologies de pointe pour offrir une expérience utilisateur exceptionnelle 
                   avec des temps de réponse optimisés et une qualité audio remarquable."""),
        ("numerique", "Les chiffres : 1, 2, 3, 10, 100, 1000, 2024, 3.14159, 50%, €123.45"),
        ("ponctuation", "Test de ponctuation : virgule, point-virgule ; point d'exclamation ! point d'interrogation ? guillemets « français » et \"anglais\"."),
        ("accents", "Café, naïve, cœur, Noël, été, français, château, hôtel, théâtre, créé."),
    ]
    
    try:
        # Initialisation
        config_path = Path("config/tts.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        tts_manager = UnifiedTTSManager(config)
        print("✅ TTS Manager initialisé")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results = []
        
        print(f"\n🔄 Génération de {len(test_cases)} fichiers de test...")
        
        for i, (test_name, text) in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}/{len(test_cases)}: {test_name}")
            print(f"   Texte: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            try:
                # Synthèse
                start_time = time.perf_counter()
                tts_result = await tts_manager.synthesize(text)
                synthesis_time = (time.perf_counter() - start_time) * 1000
                
                # Extraction audio
                if hasattr(tts_result, 'audio_data'):
                    wav_bytes = tts_result.audio_data
                else:
                    wav_bytes = tts_result
                
                # Sauvegarde
                filename = f"demo_batch_{test_name}_{timestamp}.wav"
                with open(filename, "wb") as f:
                    f.write(wav_bytes)
                
                # Validation
                is_valid = is_valid_wav(wav_bytes)
                audio_info = get_wav_info(wav_bytes)
                
                results.append({
                    'name': test_name,
                    'filename': filename,
                    'synthesis_time': synthesis_time,
                    'audio_size': len(wav_bytes),
                    'duration_ms': audio_info.get('duration_ms', 0),
                    'valid': is_valid,
                    'text_length': len(text)
                })
                
                print(f"   ✅ Généré: {filename} ({synthesis_time:.1f}ms)")
                
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
                results.append({
                    'name': test_name,
                    'filename': None,
                    'error': str(e)
                })
        
        # Rapport final
        print("\n" + "="*60)
        print("📊 RAPPORT GÉNÉRATION BATCH")
        print("="*60)
        
        successful = [r for r in results if r.get('filename')]
        failed = [r for r in results if not r.get('filename')]
        
        print(f"✅ Réussis: {len(successful)}/{len(test_cases)}")
        print(f"❌ Échecs: {len(failed)}")
        print()
        
        if successful:
            print("📁 FICHIERS GÉNÉRÉS:")
            total_time = sum(r['synthesis_time'] for r in successful)
            total_chars = sum(r['text_length'] for r in successful)
            
            for result in successful:
                print(f"   {result['filename']}")
                print(f"      {result['synthesis_time']:.1f}ms | {result['audio_size']:,} bytes | {result['duration_ms']:.0f}ms audio")
            
            print(f"\n📈 STATISTIQUES:")
            print(f"   Temps total: {total_time:.1f}ms")
            print(f"   Caractères total: {total_chars}")
            print(f"   Débit moyen: {total_chars/(total_time/1000):.1f} chars/s")
        
        if failed:
            print("\n❌ ÉCHECS:")
            for result in failed:
                print(f"   {result['name']}: {result.get('error', 'Erreur inconnue')}")
        
        print(f"\n🎧 Pour tester la qualité audio, écoutez les fichiers générés.")
        
        # Nettoyage
        await tts_manager.cleanup()
        
    except Exception as e:
        print(f"❌ Erreur démonstration batch: {e}")

async def main():
    """Point d'entrée principal"""
    print("🎵 SuperWhisper V6 - Démonstration TTS")
    print("🚀 Test manuel avec génération de fichiers audio")
    
    print("\nChoisissez le mode de démonstration :")
    print("1. Interactif (saisie manuelle)")
    print("2. Batch (génération automatique)")
    print("3. Les deux")
    
    choice = input("\nVotre choix (1/2/3) : ").strip()
    
    if choice == '1':
        await demo_tts_interactive()
    elif choice == '2':
        await demo_tts_batch()
    elif choice == '3':
        await demo_tts_batch()
        print("\n" + "="*80)
        await demo_tts_interactive()
    else:
        print("❌ Choix invalide")

if __name__ == "__main__":
    asyncio.run(main()) 