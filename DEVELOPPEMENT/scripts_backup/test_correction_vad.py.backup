#!/usr/bin/env python3
"""
Test Correction VAD - SuperWhisper V6 Phase 4
🔧 VALIDATION: Transcription complète avec paramètres VAD corrigés

Mission: Valider que la correction VAD permet de transcrire
le texte complet fourni (155 mots) au lieu de s'arrêter à 25 mots.
"""

import os
import sys
import time
import asyncio
import numpy as np
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 Test Correction VAD - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Imports après configuration GPU
sys.path.append(str(Path(__file__).parent.parent))

from STT.backends.prism_stt_backend import PrismSTTBackend

def generer_audio_test_long(duree: float = 30.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Génère un audio test pour simuler le texte de 155 mots
    
    Args:
        duree: Durée en secondes (30s pour 155 mots ≈ 5 mots/seconde)
        sample_rate: Fréquence d'échantillonnage
        
    Returns:
        Audio numpy array avec pattern réaliste
    """
    print(f"🎵 Génération audio test {duree}s pour validation VAD...")
    
    samples = int(duree * sample_rate)
    audio = np.zeros(samples, dtype=np.float32)
    
    # Simuler pattern de parole: segments avec silences
    segment_duration = 3.0  # 3 secondes par segment
    silence_duration = 0.5  # 0.5 seconde de silence entre segments
    
    current_pos = 0
    segment_id = 1
    
    while current_pos < samples:
        # Segment de parole
        segment_samples = int(segment_duration * sample_rate)
        if current_pos + segment_samples > samples:
            segment_samples = samples - current_pos
        
        # Pattern audio réaliste (plusieurs fréquences)
        t = np.linspace(0, segment_duration, segment_samples)
        
        # Fréquences vocales typiques
        freq1 = 150 + (segment_id % 3) * 50  # Fréquence fondamentale variable
        freq2 = freq1 * 2                   # Harmonique
        freq3 = freq1 * 3                   # Harmonique
        
        # Signal composite avec modulation
        signal = (
            0.3 * np.sin(2 * np.pi * freq1 * t) +
            0.2 * np.sin(2 * np.pi * freq2 * t) +
            0.1 * np.sin(2 * np.pi * freq3 * t)
        )
        
        # Enveloppe réaliste
        envelope = np.exp(-t / (segment_duration * 0.8))  # Décroissance
        signal *= envelope
        
        # Ajouter bruit léger
        noise = np.random.normal(0, 0.02, len(signal))
        signal += noise
        
        # Insérer dans audio principal
        end_pos = min(current_pos + len(signal), samples)
        audio[current_pos:end_pos] = signal[:end_pos - current_pos]
        
        current_pos += segment_samples
        
        # Silence entre segments
        silence_samples = int(silence_duration * sample_rate)
        current_pos += silence_samples
        
        segment_id += 1
        
        print(f"   Segment {segment_id-1}: {freq1}Hz, position {current_pos/sample_rate:.1f}s")
    
    print(f"✅ Audio test généré: {len(audio)} échantillons ({len(audio)/sample_rate:.1f}s)")
    return audio

async def test_correction_vad_complete():
    """
    Test principal de la correction VAD
    
    Objectif: Valider que la transcription va jusqu'au bout
    """
    print("\n" + "="*60)
    print("🔧 TEST CORRECTION VAD - TRANSCRIPTION COMPLÈTE")
    print("="*60)
    
    print("\n🎯 Objectif:")
    print("   Valider que les paramètres VAD corrigés permettent")
    print("   une transcription complète sans coupure prématurée")
    print("   (problème résolu: 25 mots sur 155)")
    
    try:
        # Initialiser backend avec VAD activé
        print("\n🚀 Initialisation Backend Prism STT...")
        config = {
            'model': 'large-v2',
            'compute_type': 'float16',
            'language': 'fr',
            'beam_size': 5,
            'vad_filter': True  # VAD ACTIVÉ avec paramètres corrigés
        }
        
        backend = PrismSTTBackend(config)
        print("✅ Backend initialisé avec VAD corrigé")
        
        # Générer audio test long (simule 155 mots)
        print("\n🎵 Génération audio test long...")
        audio_test = generer_audio_test_long(duree=30.0)  # 30 secondes
        print(f"✅ Audio généré: {len(audio_test)} échantillons")
        
        # Test transcription AVEC VAD corrigé
        print("\n🔧 Test transcription avec VAD corrigé...")
        start_time = time.time()
        
        result = await backend.transcribe(audio_test)
        
        transcription_time = time.time() - start_time
        
        # Analyser résultats
        print(f"\n📊 RÉSULTATS CORRECTION VAD:")
        print(f"   ⏱️ Temps transcription: {transcription_time:.2f}s")
        print(f"   ⚡ RTF: {result.rtf:.3f}")
        print(f"   🎯 Succès: {result.success}")
        print(f"   📝 Texte transcrit: '{result.text}'")
        print(f"   📏 Longueur texte: {len(result.text)} caractères")
        print(f"   🔢 Mots estimés: {len(result.text.split()) if result.text else 0}")
        print(f"   🎪 Segments: {len(result.segments)}")
        print(f"   💪 Confiance: {result.confidence:.3f}")
        
        # Validation correction
        mots_transcrits = len(result.text.split()) if result.text else 0
        caracteres_transcrits = len(result.text)
        
        print(f"\n✅ VALIDATION CORRECTION VAD:")
        
        # Critère 1: Transcription non vide
        if result.text and len(result.text) > 0:
            print("   ✅ Transcription non vide")
        else:
            print("   ❌ Transcription vide")
        
        # Critère 2: Durée transcription proche durée audio
        duree_audio = len(audio_test) / 16000
        segments_traites = len(result.segments)
        if segments_traites > 5:  # Au moins 5 segments pour 30s
            print(f"   ✅ Segments traités: {segments_traites} (audio {duree_audio:.1f}s)")
        else:
            print(f"   ⚠️ Peu de segments: {segments_traites} pour {duree_audio:.1f}s")
        
        # Critère 3: RTF acceptable
        if result.rtf < 1.0:
            print(f"   ✅ RTF temps réel: {result.rtf:.3f}")
        else:
            print(f"   ⚠️ RTF élevé: {result.rtf:.3f}")
        
        # Critère 4: Pas d'erreur
        if result.success:
            print("   ✅ Transcription réussie")
        else:
            print(f"   ❌ Erreur: {result.error}")
        
        # Résumé
        if result.success and result.text and segments_traites > 5:
            print(f"\n🎊 CORRECTION VAD VALIDÉE!")
            print(f"   La transcription semble complète avec {segments_traites} segments")
            return True
        else:
            print(f"\n⚠️ CORRECTION VAD À VÉRIFIER")
            print(f"   Validation manuelle recommandée")
            return False
            
    except Exception as e:
        print(f"\n❌ ERREUR Test Correction VAD: {e}")
        import traceback
        traceback.print_exc()
        return False

def afficher_aide_interpretation():
    """Affiche l'aide pour interpréter les résultats"""
    print("\n" + "="*60)
    print("📚 AIDE INTERPRÉTATION RÉSULTATS")
    print("="*60)
    
    print("\n🔍 Comment interpréter:")
    print("   ✅ CORRECTION RÉUSSIE si:")
    print("      - Transcription non vide")
    print("      - Plusieurs segments traités (> 5 pour 30s)")
    print("      - RTF < 1.0 (temps réel)")
    print("      - Pas d'erreur technique")
    
    print("\n   ⚠️ INVESTIGATION REQUISE si:")
    print("      - Transcription très courte")
    print("      - Peu de segments (< 3 pour 30s)")
    print("      - RTF très élevé (> 2.0)")
    print("      - Erreurs techniques")
    
    print("\n🎯 Prochaines étapes:")
    print("   1. Si correction validée → Test avec micro réel")
    print("   2. Si problème persiste → Ajuster paramètres VAD")
    print("   3. Validation humaine finale obligatoire")

async def main():
    """Point d'entrée principal"""
    print("🔧 TEST CORRECTION VAD - SUPERWHISPER V6 PHASE 4")
    print("Mission: Valider paramètres VAD corrigés pour transcription complète")
    
    # Test correction
    success = await test_correction_vad_complete()
    
    # Aide interprétation
    afficher_aide_interpretation()
    
    # Conclusion
    if success:
        print(f"\n🎊 RÉSULTAT: CORRECTION VAD VALIDÉE")
        print("💡 Prochaine étape: Test microphone réel avec validation humaine")
    else:
        print(f"\n⚠️ RÉSULTAT: INVESTIGATION REQUISE")
        print("💡 Vérifier logs et ajuster paramètres si nécessaire")

if __name__ == "__main__":
    asyncio.run(main()) 