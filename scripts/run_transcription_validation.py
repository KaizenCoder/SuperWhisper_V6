#!/usr/bin/env python3
"""
Benchmark de Transcription Haute Fidélité - SuperWhisper V6

Ce script lance une session de benchmark, calcule les métriques de performance
(WER) et opérationnelles (VAD, RMS) et sauvegarde le tout dans un
rapport JSON structuré.
"""

import os
import sys
import asyncio
import argparse
import json
from pathlib import Path
from jiwer import wer
import time

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ajout du chemin racine au sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from STT.streaming_microphone_manager import StreamingMicrophoneManager
from STT.unified_stt_manager import UnifiedSTTManager

# Texte de référence pour le calcul du WER
REFERENCE_TEXT = """
Bonjour, ceci est un test de validation pour SuperWhisper 2.
Je vais maintenant énoncer plusieurs phrases de complexité croissante pour évaluer la précision de transcription.
Premièrement, des mots simples : chat, chien, maison, voiture, ordinateur, téléphone.
Deuxièmement, des phrases courtes : Il fait beau aujourd'hui. Le café est délicieux. J'aime la musique classique.
Troisièmement, des phrases plus complexes : L'intelligence artificielle transforme notre manière de travailler et de communiquer dans le monde moderne.
Quatrièmement, des termes techniques : algorithme, machine learning, GPU RTX 3090, faster-whisper, quantification INT8, latence de transcription.
Cinquièmement, des nombres et dates : vingt-trois, quarante-sept, mille neuf cent quatre-vingt-quinze, le quinze janvier deux mille vingt-quatre.
Sixièmement, des mots difficiles : chrysanthème, anticonstitutionnellement, prestidigitateur, kakémono, yaourt.
Septièmement, une phrase longue et complexe : L'optimisation des performances de transcription vocale nécessite une approche méthodique combinant la sélection appropriée des modèles, l'ajustement des paramètres de traitement, et l'implémentation d'algorithmes de post-traitement pour améliorer la qualité du résultat final.
Fin du test de validation.
""".strip().lower()

def get_stt_config(model_name: str) -> dict:
    """Génère la configuration pour le UnifiedSTTManager."""
    return {
        'backends': [
            {'name': f'prism_{model_name}', 'type': 'prism', 'model': model_name}
        ],
        'fallback_chain': [f'prism_{model_name}'],
        'cache_size_mb': 50,
        'cache_ttl': 3600,
        'timeout_per_minute': 10.0
    }

async def main(model_name: str, output_file: str):
    """
    Fonction principale pour exécuter le benchmark.
    """
    print("="*60)
    print("🚀 BENCHMARK DE TRANSCRIPTION HAUTE FIDÉLITÉ - SUPERWHISPER V6")
    print(f"   Modèle utilisé : {model_name}")
    print(f"   Rapport JSON   : {output_file}")
    print("="*60)

    streaming_manager = None
    try:
        # Initialisation du manager STT
        stt_config = get_stt_config(model_name)
        stt_manager = UnifiedSTTManager(config=stt_config)
        
        # Fichier temporaire pour la transcription brute
        temp_transcription_path = Path(output_file).parent / "temp_transcription.txt"

        # Initialisation du manager de streaming
        streaming_manager = StreamingMicrophoneManager(
            stt_manager=stt_manager, 
            model_name=model_name,
            output_file=str(temp_transcription_path)
        )

        print("\n🎤 Le système est prêt. L'enregistrement commencera dès que vous parlerez.")
        print("🗣️  Veuillez lire le texte de référence maintenant.")
        print("   (Appuyez sur Ctrl+C pour arrêter la transcription)")
        print("-"*60)
        
        # Démarrer la tâche de streaming
        streaming_task = asyncio.create_task(streaming_manager.run())
        await streaming_task

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n✅ Transcription interrompue par l'utilisateur.")
    
    except Exception as e:
        print(f"\n❌ Une erreur est survenue : {e}")
        import traceback
        traceback.print_exc()
        return # Ne pas générer de rapport en cas d'erreur
    
    finally:
        if streaming_manager:
            print("\n📊 Génération du rapport de benchmark...")

            # Lire la transcription brute
            try:
                with open(temp_transcription_path, 'r', encoding='utf-8') as f:
                    transcribed_text = f.read().replace('\n', ' ').strip().lower()
                temp_transcription_path.unlink() # Supprimer le fichier temporaire
            except FileNotFoundError:
                transcribed_text = ""
                print("⚠️ Fichier de transcription temporaire non trouvé.")

            # Calcul du WER
            error_rate = wer(REFERENCE_TEXT, transcribed_text)

            # Récupération des métriques opérationnelles
            op_metrics = streaming_manager.get_final_stats()

            # Création du rapport JSON
            report = {
                "metadata": {
                    "model_used": model_name,
                    "test_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                },
                "reference_text": REFERENCE_TEXT,
                "transcribed_text": transcribed_text,
                "performance_metrics": {
                    "word_error_rate_percent": round(error_rate * 100, 2)
                },
                "operational_metrics": {
                    "chunks_processed": int(op_metrics.get("chunks_processed", 0)),
                    "chunks_with_voice": int(op_metrics.get("chunks_with_voice", 0)),
                    "chunks_filtered_noise": int(op_metrics.get("chunks_filtered_noise", 0)),
                    "hallucinations_detected": int(op_metrics.get("hallucinations_detected", 0)),
                    "transcribed_segments": int(op_metrics.get("transcribed_segments", 0)),
                    "average_rms": float(op_metrics.get("average_rms", 0.0)),
                    "vad_efficiency_percent": float(op_metrics.get("vad_efficiency_percent", 0.0))
                }
            }

            # Sauvegarde du rapport
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            
            print(f"📄 Rapport de benchmark complet sauvegardé dans : {output_file}")
            print(f"🎯 Taux d'erreur (WER): {report['performance_metrics']['word_error_rate_percent']}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de benchmark de transcription SuperWhisper V6.")
    parser.add_argument(
        '--model', 
        type=str, 
        default='large-v2', 
        choices=['small', 'medium', 'large-v2'],
        help="Modèle Whisper à utiliser pour le benchmark."
    )
    args = parser.parse_args()
    
    # Créer le chemin de sortie dans un dossier 'results' s'il n'existe pas
    output_path = Path('results')
    output_path.mkdir(exist_ok=True)
    
    # Nom du fichier de rapport dynamique
    report_filename = f"benchmark_report_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    full_output_path = output_path / report_filename

    asyncio.run(main(args.model, str(full_output_path))) 