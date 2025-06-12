#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test TTS avec Lecture Audio Automatique - SuperWhisper V6
Script qui génère ET joue l'audio pour validation auditive immédiate
"""

import os
import sys
import asyncio
import time
import subprocess
from pathlib import Path

# Configuration GPU RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Import du système TTS
try:
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from TTS.tts_manager import UnifiedTTSManager
    from TTS.utils_audio import is_valid_wav, get_wav_info
    import yaml
    TTS_AVAILABLE = True
    print("✅ Système TTS disponible")
except ImportError as e:
    print(f"❌ Système TTS non disponible: {e}")
    TTS_AVAILABLE = False

def play_audio_file(filepath):
    """Joue un fichier audio avec le lecteur par défaut de Windows"""
    try:
        # Utilise le lecteur par défaut de Windows
        subprocess.run(['start', '', str(filepath)], shell=True, check=True)
        print(f"🔊 Lecture audio lancée: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Erreur lecture audio: {e}")
        return False

async def test_avec_ecoute():
    """Test TTS avec écoute immédiate"""
    if not TTS_AVAILABLE:
        print("❌ Système TTS non disponible")
        return False
    
    try:
        # Initialisation
        config_path = Path("config/tts.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        tts_manager = UnifiedTTSManager(config)
        print("✅ TTS Manager initialisé")
        
        # Tests avec écoute
        test_cases = [
            ("Test court", "Bonjour, ceci est un test de synthèse vocale SuperWhisper V6."),
            ("Test qualité", "La qualité audio est-elle satisfaisante ? Pouvez-vous entendre clairement cette phrase ?"),
            ("Test accents", "Café, naïve, cœur, Noël, été, français, château, hôtel."),
            ("Test numérique", "Les chiffres : 1, 2, 3, 10, 100, 1000, 2024, 3.14159."),
        ]
        
        print("\n🎵 TESTS AVEC ÉCOUTE AUTOMATIQUE")
        print("=" * 50)
        
        for i, (nom, texte) in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}/{len(test_cases)}: {nom}")
            print(f"   Texte: {texte[:60]}...")
            
            # Génération audio
            start_time = time.perf_counter()
            tts_result = await tts_manager.synthesize(texte)
            synthesis_time = (time.perf_counter() - start_time) * 1000
            
            # Extraction audio
            if hasattr(tts_result, 'audio_data'):
                wav_bytes = tts_result.audio_data
            else:
                wav_bytes = tts_result
            
            if wav_bytes is None:
                print(f"   ❌ Échec génération audio")
                continue
            
            # Sauvegarde
            filename = f"test_ecoute_{i}_{nom.lower().replace(' ', '_')}.wav"
            with open(filename, "wb") as f:
                f.write(wav_bytes)
            
            # Validation
            is_valid = is_valid_wav(wav_bytes)
            audio_info = get_wav_info(wav_bytes)
            
            print(f"   ✅ Généré: {synthesis_time:.1f}ms, {len(wav_bytes)} bytes")
            print(f"   📊 Audio: {audio_info['duration']:.1f}s, {audio_info['sample_rate']}Hz")
            
            # LECTURE AUTOMATIQUE
            print(f"   🔊 Lecture automatique en cours...")
            play_audio_file(filename)
            
            # Pause pour écoute
            print(f"   ⏳ Pause 3 secondes pour écoute...")
            await asyncio.sleep(3)
        
        # Nettoyage
        await tts_manager.cleanup()
        
        print("\n" + "=" * 50)
        print("🎉 Tests avec écoute terminés !")
        print("💡 Les fichiers audio restent disponibles pour réécoute.")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Point d'entrée principal"""
    print("🎵 SuperWhisper V6 - Test TTS avec Écoute Automatique")
    print("=" * 60)
    print("🔊 Ce script va générer ET jouer l'audio automatiquement")
    print("🎧 Assurez-vous que vos haut-parleurs/casque sont allumés")
    
    input("\n▶️  Appuyez sur Entrée pour commencer les tests audio...")
    
    # Test avec écoute
    success = asyncio.run(test_avec_ecoute())
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Tests audio réussis ! Avez-vous entendu les synthèses ?")
    else:
        print("❌ Échec des tests audio")
    
    input("\n⏸️  Appuyez sur Entrée pour terminer...")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 