#!/usr/bin/env python3
"""
🎯 TEST STREAMING MICROPHONE AVEC TEXTE DE RÉFÉRENCE
Validation précise avec le texte de référence de 155 mots

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Ajouter le répertoire parent au PYTHONPATH pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Imports après configuration GPU
try:
    import torch
    from STT.unified_stt_manager import UnifiedSTTManager
    from STT.streaming_microphone_manager import StreamingMicrophoneManager, SpeechSegment
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    sys.exit(1)

# =============================================================================
# 📝 TEXTE DE RÉFÉRENCE - 155 MOTS À TRANSCRIRE INTÉGRALEMENT
# =============================================================================
TEXTE_REFERENCE_155_MOTS = """
Dans le cadre du développement de SuperWhisper V6, nous procédons à l'intégration du module Speech-to-Text 
utilisant Prism_Whisper2 optimisé pour la configuration RTX 3090. Cette phase critique nécessite une 
validation rigoureuse des paramètres Voice Activity Detection pour assurer une transcription complète 
et précise. Le système doit être capable de traiter des phrases complexes, techniques et longues sans 
interruption prématurée. L'architecture implémentée comprend un gestionnaire unifié avec fallback 
automatique, un cache LRU optimisé et des circuit breakers pour la robustesse. Les performances ciblées 
incluent une latence inférieure à quatre cents millisecondes pour cinq secondes d'audio et un facteur 
temps réel inférieur à un. La configuration GPU utilise exclusivement la RTX 3090 via CUDA_VISIBLE_DEVICES 
pour garantir l'allocation mémoire optimale. Ce test de validation doit confirmer que tous ces éléments 
fonctionnent harmonieusement ensemble pour produire une transcription fidèle et complète du texte prononcé.
""".strip()

# =============================================================================
# VALIDATION RTX 3090 OBLIGATOIRE
# =============================================================================
def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

# =============================================================================
# ANALYSEUR DE PRÉCISION TEXTE
# =============================================================================
@dataclass
class AnalysePrecision:
    """Résultats d'analyse de précision"""
    mots_reference: int
    mots_transcrits: int
    mots_corrects: int
    precision_pourcentage: float
    couverture_pourcentage: float
    mots_manques: List[str]
    mots_ajoutes: List[str]

def analyser_precision_texte(texte_reference: str, texte_transcrit: str) -> AnalysePrecision:
    """Analyse précise de la transcription vs référence"""
    
    # Normalisation
    def normaliser(texte):
        return texte.lower().replace(',', '').replace('.', '').replace(':', '').replace(';', '').strip()
    
    ref_norm = normaliser(texte_reference)
    trans_norm = normaliser(texte_transcrit)
    
    mots_ref = ref_norm.split()
    mots_trans = trans_norm.split()
    
    # Calculs
    mots_ref_set = set(mots_ref)
    mots_trans_set = set(mots_trans)
    
    mots_corrects_set = mots_ref_set.intersection(mots_trans_set)
    mots_manques = list(mots_ref_set - mots_trans_set)
    mots_ajoutes = list(mots_trans_set - mots_ref_set)
    
    precision = len(mots_corrects_set) / len(mots_ref_set) * 100 if mots_ref_set else 0
    couverture = len(mots_trans) / len(mots_ref) * 100 if mots_ref else 0
    
    return AnalysePrecision(
        mots_reference=len(mots_ref),
        mots_transcrits=len(mots_trans),
        mots_corrects=len(mots_corrects_set),
        precision_pourcentage=precision,
        couverture_pourcentage=couverture,
        mots_manques=mots_manques[:10],  # Limiter affichage
        mots_ajoutes=mots_ajoutes[:10]
    )

# =============================================================================
# GESTIONNAIRE TEST STREAMING AVEC RÉFÉRENCE
# =============================================================================
class TestStreamingReference:
    """Test streaming microphone avec validation référence"""
    
    def __init__(self):
        validate_rtx3090_configuration()
        
        self.stt_manager = None
        self.mic_manager = None
        self.segments_transcrits = []
        self.texte_complet = ""
        self.debut_test = None
        
    async def initialize(self):
        """Initialisation du test"""
        print("\n🚀 INITIALISATION TEST STREAMING RÉFÉRENCE")
        print("="*60)
        
        # STT Manager
        print("\n🎤 Initialisation STT Manager...")
        stt_config = {
            'timeout_per_minute': 10.0,
            'max_retries': 3,
            'cache_enabled': True,
            'circuit_breaker_enabled': True,
            'fallback_chain': ['prism_primary'],
            'backends': [
                {
                    'name': 'prism_primary',
                    'type': 'prism',
                    'model': 'large-v2',
                    'compute_type': 'float16',
                    'language': 'fr',
                    'beam_size': 5,
                    'vad_filter': True
                }
            ]
        }
        
        self.stt_manager = UnifiedSTTManager(config=stt_config)
        print("✅ STT Manager initialisé")
        
        # Streaming Microphone
        print("\n🎙️ Initialisation Streaming Microphone...")
        self.mic_manager = StreamingMicrophoneManager(
            stt_manager=self.stt_manager,
            on_transcription=self.on_segment_transcrit
        )
        print("✅ Streaming Microphone initialisé")
        
        print("\n🎉 TEST STREAMING RÉFÉRENCE PRÊT")
        print("="*60)
    
    def on_segment_transcrit(self, text: str, segment: SpeechSegment):
        """Callback appelé pour chaque segment transcrit"""
        if not text.strip():
            return
        
        self.segments_transcrits.append({
            'texte': text,
            'segment': segment,
            'timestamp': time.time()
        })
        
        self.texte_complet += " " + text
        
        print(f"\n🗣️ SEGMENT {len(self.segments_transcrits)}")
        print(f"   📝 Texte: '{text}'")
        print(f"   ⏱️ Durée: {segment.duration_ms}ms")
        print(f"   🚀 Latence: {int((time.time() - segment.end_ts) * 1000)}ms")
        print(f"   📊 Mots segment: {len(text.split())}")
        print(f"   📈 Total mots: {len(self.texte_complet.split())}")
    
    async def run_test_reference(self):
        """Test avec texte de référence"""
        print("\n🎯 TEST STREAMING MICROPHONE - TEXTE DE RÉFÉRENCE")
        print("="*70)
        
        # Affichage texte référence
        nb_mots_ref = len(TEXTE_REFERENCE_155_MOTS.split())
        print(f"\n📋 TEXTE DE RÉFÉRENCE À LIRE:")
        print(f"   📏 Longueur: {len(TEXTE_REFERENCE_155_MOTS)} caractères")
        print(f"   🔢 Nombre de mots: {nb_mots_ref}")
        print(f"   ⏱️ Durée estimée: {nb_mots_ref * 0.4:.1f}s (150 mots/min)")
        
        print(f"\n📝 CONTENU À LIRE:")
        print("-" * 50)
        print(TEXTE_REFERENCE_155_MOTS)
        print("-" * 50)
        
        print(f"\n📋 INSTRUCTIONS:")
        print(f"   1. Lisez le texte COMPLET à voix haute")
        print(f"   2. Articulez clairement mais naturellement")
        print(f"   3. Respectez la ponctuation (pauses)")
        print(f"   4. Ne vous arrêtez pas en cours de lecture")
        print(f"   5. Appuyez Ctrl+C quand vous avez terminé")
        
        input(f"\n🎤 Appuyez sur ENTRÉE quand vous êtes prêt à commencer...")
        
        print(f"\n🔴 STREAMING DÉMARRÉ - LISEZ LE TEXTE MAINTENANT")
        print(f"🛑 Appuyez Ctrl+C quand vous avez terminé la lecture")
        print("-" * 60)
        
        self.debut_test = time.time()
        
        try:
            await self.mic_manager.run()
        except KeyboardInterrupt:
            print(f"\n🛑 Test arrêté par utilisateur")
        finally:
            await self.analyser_resultats()
    
    async def analyser_resultats(self):
        """Analyse des résultats vs référence"""
        duree_test = time.time() - self.debut_test if self.debut_test else 0
        
        print(f"\n📊 ANALYSE RÉSULTATS TEST RÉFÉRENCE")
        print("="*60)
        
        # Analyse précision
        analyse = analyser_precision_texte(TEXTE_REFERENCE_155_MOTS, self.texte_complet.strip())
        
        print(f"\n🎯 MÉTRIQUES PRÉCISION:")
        print(f"   📝 Mots référence: {analyse.mots_reference}")
        print(f"   📝 Mots transcrits: {analyse.mots_transcrits}")
        print(f"   ✅ Mots corrects: {analyse.mots_corrects}")
        print(f"   🎯 Précision: {analyse.precision_pourcentage:.1f}%")
        print(f"   📊 Couverture: {analyse.couverture_pourcentage:.1f}%")
        
        # Métriques streaming
        print(f"\n⏱️ MÉTRIQUES STREAMING:")
        print(f"   🗣️ Segments traités: {len(self.segments_transcrits)}")
        print(f"   ⏰ Durée test: {duree_test:.1f}s")
        print(f"   📊 Segments/minute: {len(self.segments_transcrits) / (duree_test/60):.1f}")
        
        if self.segments_transcrits:
            latences = [int((seg['timestamp'] - seg['segment'].end_ts) * 1000) for seg in self.segments_transcrits]
            latence_moy = sum(latences) / len(latences)
            print(f"   🚀 Latence moyenne: {latence_moy:.0f}ms")
        
        # Analyse qualitative
        print(f"\n🔍 ANALYSE QUALITATIVE:")
        
        if analyse.precision_pourcentage >= 90:
            print(f"   ✅ EXCELLENT: Précision ≥ 90%")
        elif analyse.precision_pourcentage >= 80:
            print(f"   ✅ BON: Précision ≥ 80%")
        elif analyse.precision_pourcentage >= 70:
            print(f"   ⚠️ MOYEN: Précision ≥ 70%")
        else:
            print(f"   ❌ FAIBLE: Précision < 70%")
        
        if analyse.couverture_pourcentage >= 95:
            print(f"   ✅ COUVERTURE COMPLÈTE: ≥ 95%")
        elif analyse.couverture_pourcentage >= 85:
            print(f"   ✅ BONNE COUVERTURE: ≥ 85%")
        else:
            print(f"   ⚠️ COUVERTURE PARTIELLE: {analyse.couverture_pourcentage:.1f}%")
        
        # Mots manqués/ajoutés
        if analyse.mots_manques:
            print(f"\n❌ MOTS MANQUÉS (échantillon):")
            print(f"   {', '.join(analyse.mots_manques)}")
        
        if analyse.mots_ajoutes:
            print(f"\n➕ MOTS AJOUTÉS (échantillon):")
            print(f"   {', '.join(analyse.mots_ajoutes)}")
        
        # Transcription complète
        print(f"\n📝 TRANSCRIPTION COMPLÈTE:")
        print("-" * 50)
        print(self.texte_complet.strip())
        print("-" * 50)
        
        # Verdict final
        print(f"\n🏆 VERDICT FINAL:")
        if analyse.precision_pourcentage >= 85 and analyse.couverture_pourcentage >= 90:
            print(f"   ✅ VALIDATION RÉUSSIE - Streaming microphone opérationnel")
        elif analyse.precision_pourcentage >= 75 and analyse.couverture_pourcentage >= 80:
            print(f"   ⚠️ VALIDATION PARTIELLE - Améliorations possibles")
        else:
            print(f"   ❌ VALIDATION ÉCHOUÉE - Optimisations requises")

# =============================================================================
# MAIN
# =============================================================================
async def main():
    """Point d'entrée principal"""
    
    print("🎯 TEST STREAMING MICROPHONE - TEXTE DE RÉFÉRENCE 155 MOTS")
    print("="*70)
    print("🎮 GPU: RTX 3090 exclusif")
    print("🎙️ Microphone: RODE NT-USB détection automatique")
    print("📝 Validation: Texte de référence précis")
    print()
    
    try:
        # Initialisation test
        test = TestStreamingReference()
        await test.initialize()
        
        # Exécution test avec référence
        await test.run_test_reference()
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 SuperWhisper V6 - Test Streaming avec Texte de Référence")
    print("📝 Validation précise avec 155 mots de référence")
    print()
    
    asyncio.run(main()) 