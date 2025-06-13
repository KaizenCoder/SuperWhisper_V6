#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de performance TTS Piper avec RTX 3090
- Configuration GPU correcte
- Résolution DLLs CUDA
- Benchmark de performance
"""

import sys
import os
import time
import traceback

# Configuration RTX 3090 AVANT tous les imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ajouter le répertoire PyTorch au PATH pour les DLLs CUDA
torch_lib_path = os.path.join(os.getcwd(), 'venv_piper312', 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib_path):
    current_path = os.environ.get('PATH', '')
    if torch_lib_path not in current_path:
        os.environ['PATH'] = current_path + os.pathsep + torch_lib_path
        print(f"✅ DLLs CUDA ajoutées au PATH: {torch_lib_path}")

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rtx3090_performance():
    """Test de performance TTS complet RTX 3090"""
    
    print("🚀 TEST PERFORMANCE TTS RTX 3090")
    print("=" * 60)
    
    # Configuration du test
    config = {
        'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
        'use_gpu': True,
        'sample_rate': 22050,
        'noise_scale': 0.667,
        'noise_scale_w': 0.8,
        'length_scale': 1.0
    }
    
    try:
        # Vérification du modèle
        print("\\n1. 🔍 Vérification du modèle...")
        if not os.path.exists(config['model_path']):
            print(f"❌ Modèle non trouvé: {config['model_path']}")
            return False
            
        model_size = os.path.getsize(config['model_path']) / 1024**2
        print(f"✅ Modèle trouvé: {config['model_path']} ({model_size:.1f} MB)")
        
        # Vérification PyTorch + RTX 3090
        print("\\n2. 🎮 Vérification RTX 3090...")
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)  # RTX 3090 (CUDA:0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # RTX 3090 (CUDA:0)
            print(f"✅ GPU détecté: {gpu_name} ({gpu_memory:.1f}GB)")
            
            if "RTX 3090" in gpu_name or gpu_memory >= 20:
                print("✅ RTX 3090 confirmé !")
            else:
                print("⚠️ GPU différent du RTX 3090 attendu")
        else:
            print("❌ CUDA non disponible")
            
        # Import du handler
        print("\\n3. 📦 Import du handler RTX 3090...")
        from TTS.tts_handler_piper_rtx3090 import TTSHandlerPiperRTX3090
        print("✅ Handler RTX 3090 importé")
        
        # Initialisation
        print("\\n4. 🚀 Initialisation RTX 3090...")
        start_init = time.time()
        handler = TTSHandlerPiperRTX3090(config)
        init_time = time.time() - start_init
        print(f"✅ Handler RTX 3090 initialisé en {init_time:.2f}s")
        
        # Tests de performance
        test_texts = [
            "Salut !",
            "Bonjour, comment allez-vous ?",
            "LUXA est un assistant vocal intelligent développé avec des technologies d'IA avancées.",
            "Ceci est un test de synthèse vocale française avec Piper sur RTX 3090 pour mesurer les performances GPU et optimiser la vitesse de génération audio."
        ]
        
        print("\\n5. 🏃‍♂️ Benchmark de performance RTX 3090...")
        
        total_chars = 0
        total_time = 0
        
        for i, text in enumerate(test_texts):
            print(f"\\n   Test {i+1}/4: \"{text}\"")
            print(f"   📝 Longueur: {len(text)} caractères")
            
            start_time = time.time()
            audio_data = handler.synthesize(text)
            synth_time = time.time() - start_time
            
            if len(audio_data) > 0:
                chars_per_sec = len(text) / synth_time
                audio_duration = len(audio_data) / config['sample_rate']
                rtf = synth_time / audio_duration  # Real-time factor
                
                print(f"   ⚡ Temps: {synth_time:.3f}s")
                print(f"   🎵 Audio: {len(audio_data)} échantillons ({audio_duration:.2f}s)")
                print(f"   📊 Performance: {chars_per_sec:.1f} char/s")
                print(f"   🚀 RTF: {rtf:.3f} (plus petit = plus rapide)")
                
                total_chars += len(text)
                total_time += synth_time
                
                # Lecture du dernier test
                if i == len(test_texts) - 1:
                    print(f"   🔊 Lecture audio test...")
                    handler.speak(text)
            else:
                print(f"   ❌ Échec synthèse")
                
        # Résumé des performances
        if total_time > 0:
            avg_chars_per_sec = total_chars / total_time
            print(f"\\n6. 📊 RÉSUMÉ PERFORMANCE RTX 3090:")
            print(f"   📝 Total caractères: {total_chars}")
            print(f"   ⏱️ Temps total: {total_time:.2f}s")
            print(f"   ⚡ Performance moyenne: {avg_chars_per_sec:.1f} caractères/s")
            
            # Évaluation des performances
            if avg_chars_per_sec >= 100:
                print(f"   🏆 EXCELLENT ! Performance GPU optimale")
            elif avg_chars_per_sec >= 50:
                print(f"   ✅ BON ! Performance acceptable")
            else:
                print(f"   ⚠️ MOYEN. Possibles améliorations nécessaires")
                
            return True
        else:
            print(f"\\n❌ Aucun test de performance réussi")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test RTX 3090: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 BENCHMARK TTS PIPER RTX 3090")
    print("Configuration: CUDA_VISIBLE_DEVICES='0' + DLLs CUDA")
    print()
    
    success = test_rtx3090_performance()
    
    print("\\n" + "=" * 60)
    if success:
        print("🎉 TEST RTX 3090 RÉUSSI - PERFORMANCES MESURÉES")
    else:
        print("🚨 ÉCHEC TEST RTX 3090 - VÉRIFIER CONFIGURATION")
    print("=" * 60) 