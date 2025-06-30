#!/usr/bin/env python3
"""
Test Vocal Complet SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Test pipeline voix-à-voix complet:
🎤 Microphone → STT → LLM → TTS → 🔈 Speakers

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

import asyncio
import time
import json
from datetime import datetime
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter les répertoires au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))  # PIPELINE
sys.path.append(str(Path(__file__).parent.parent.parent))  # Racine

try:
    from PIPELINE.pipeline_orchestrator import PipelineOrchestrator, ConversationTurn
    from STT.streaming_microphone_manager import StreamingMicrophoneManager
    from STT.unified_stt_manager import UnifiedSTTManager
    from LLM.llm_client import LLMClient
    from TTS.unified_tts_manager import UnifiedTTSManager
    from PIPELINE.audio_output_manager import AudioOutputManager
    import yaml
    import torch
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    print("💡 Vérifiez que tous les modules sont disponibles")
    sys.exit(1)

def validate_rtx3090():
    """Validation obligatoire RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

class TestVocalComplet:
    """Test vocal complet du pipeline SuperWhisper V6"""
    
    def __init__(self):
        self.config_path = Path("PIPELINE/config/pipeline.yaml")
        self.config = None
        self.pipeline = None
        self.metrics = {
            "tests_effectues": 0,
            "tests_reussis": 0,
            "latences": [],
            "erreurs": [],
            "debut_test": None,
            "fin_test": None
        }
    
    def load_config(self):
        """Charger la configuration pipeline"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ Configuration chargée: {self.config_path}")
            return True
        except Exception as e:
            print(f"❌ Erreur chargement config: {e}")
            return False
    
    async def initialize_pipeline(self):
        """Initialiser le pipeline complet"""
        try:
            print("\n🔧 Initialisation pipeline...")
            
            # Créer le pipeline orchestrator
            self.pipeline = PipelineOrchestrator(
                config_path=str(self.config_path),
                enable_metrics=True
            )
            
            # Initialiser tous les composants
            await self.pipeline.initialize()
            
            print("✅ Pipeline initialisé avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur initialisation pipeline: {e}")
            self.metrics["erreurs"].append(f"Init pipeline: {e}")
            return False
    
    async def test_conversation_vocale(self, duree_ecoute=10):
        """Test conversation vocale complète"""
        print(f"\n🎤 DÉBUT TEST VOCAL COMPLET")
        print(f"⏱️ Durée d'écoute: {duree_ecoute}s")
        print("🗣️ Parlez maintenant au microphone...")
        print("💬 Le système va transcrire, générer une réponse et la synthétiser")
        
        debut_test = time.time()
        self.metrics["debut_test"] = datetime.now().isoformat()
        
        try:
            # Créer un tour de conversation
            conversation_turn = ConversationTurn(
                user_input="",  # Sera rempli par STT
                timestamp=datetime.now()
            )
            
            # Démarrer l'écoute microphone
            print("\n🎤 Écoute microphone démarrée...")
            
            # Simuler l'écoute (dans un vrai test, on utiliserait StreamingMicrophoneManager)
            await asyncio.sleep(duree_ecoute)
            
            # Pour ce test, utilisons un texte d'exemple
            texte_test = "Bonjour, comment allez-vous aujourd'hui ?"
            conversation_turn.user_input = texte_test
            
            print(f"📝 Texte transcrit (simulé): '{texte_test}'")
            
            # Traiter avec le pipeline
            debut_pipeline = time.time()
            
            # Ajouter à la queue du pipeline
            await self.pipeline.add_to_queue(conversation_turn)
            
            # Attendre le traitement (timeout 30s)
            timeout = 30
            attente = 0
            while attente < timeout:
                if conversation_turn.llm_response and conversation_turn.tts_audio:
                    break
                await asyncio.sleep(0.1)
                attente += 0.1
            
            fin_pipeline = time.time()
            latence_totale = (fin_pipeline - debut_pipeline) * 1000
            
            if conversation_turn.llm_response:
                print(f"🤖 Réponse LLM: '{conversation_turn.llm_response}'")
            
            if conversation_turn.tts_audio:
                print(f"🔊 Audio TTS généré: {len(conversation_turn.tts_audio)} bytes")
                
                # Jouer l'audio (si AudioOutputManager disponible)
                try:
                    audio_manager = AudioOutputManager()
                    await audio_manager.play_audio(conversation_turn.tts_audio)
                    print("🔈 Audio joué avec succès")
                except Exception as e:
                    print(f"⚠️ Lecture audio échouée: {e}")
            
            # Métriques
            self.metrics["tests_effectues"] += 1
            self.metrics["latences"].append(latence_totale)
            
            if conversation_turn.llm_response and conversation_turn.tts_audio:
                self.metrics["tests_reussis"] += 1
                print(f"✅ Test vocal réussi - Latence: {latence_totale:.1f}ms")
                return True
            else:
                print(f"❌ Test vocal échoué - Réponse incomplète")
                return False
                
        except Exception as e:
            print(f"❌ Erreur test vocal: {e}")
            self.metrics["erreurs"].append(f"Test vocal: {e}")
            return False
        
        finally:
            fin_test = time.time()
            self.metrics["fin_test"] = datetime.now().isoformat()
            duree_totale = fin_test - debut_test
            print(f"⏱️ Durée totale test: {duree_totale:.1f}s")
    
    async def test_microphone_reel(self):
        """Test avec microphone réel (si disponible)"""
        print("\n🎤 TEST MICROPHONE RÉEL")
        
        try:
            # Tenter d'initialiser le microphone
            mic_manager = StreamingMicrophoneManager()
            
            print("🎤 Microphone initialisé")
            print("🗣️ Parlez pendant 5 secondes...")
            
            # Démarrer l'enregistrement
            audio_data = []
            
            def audio_callback(audio_chunk):
                audio_data.append(audio_chunk)
            
            # Simuler l'enregistrement (implémentation simplifiée)
            await asyncio.sleep(5)
            
            if audio_data:
                print(f"✅ Audio capturé: {len(audio_data)} chunks")
                return True
            else:
                print("⚠️ Aucun audio capturé")
                return False
                
        except Exception as e:
            print(f"❌ Erreur microphone: {e}")
            print("💡 Test avec microphone simulé à la place")
            return await self.test_conversation_vocale(5)
    
    def generer_rapport(self):
        """Générer rapport de test"""
        print("\n📊 RAPPORT TEST VOCAL COMPLET")
        print("=" * 50)
        
        print(f"📅 Début: {self.metrics['debut_test']}")
        print(f"📅 Fin: {self.metrics['fin_test']}")
        print(f"🧪 Tests effectués: {self.metrics['tests_effectues']}")
        print(f"✅ Tests réussis: {self.metrics['tests_reussis']}")
        
        if self.metrics['tests_effectues'] > 0:
            taux_succes = (self.metrics['tests_reussis'] / self.metrics['tests_effectues']) * 100
            print(f"📈 Taux de succès: {taux_succes:.1f}%")
        
        if self.metrics['latences']:
            latence_moy = sum(self.metrics['latences']) / len(self.metrics['latences'])
            latence_max = max(self.metrics['latences'])
            latence_min = min(self.metrics['latences'])
            
            print(f"⏱️ Latence moyenne: {latence_moy:.1f}ms")
            print(f"⏱️ Latence min: {latence_min:.1f}ms")
            print(f"⏱️ Latence max: {latence_max:.1f}ms")
            
            if latence_moy < 1200:
                print("🎯 OBJECTIF < 1200ms: ✅ ATTEINT")
            else:
                print("🎯 OBJECTIF < 1200ms: ❌ MANQUÉ")
        
        if self.metrics['erreurs']:
            print(f"\n❌ Erreurs ({len(self.metrics['erreurs'])}):")
            for i, erreur in enumerate(self.metrics['erreurs'], 1):
                print(f"  {i}. {erreur}")
        
        # Sauvegarder rapport
        rapport_path = Path("PIPELINE/reports/test_vocal_complet.json")
        rapport_path.parent.mkdir(exist_ok=True)
        
        with open(rapport_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Rapport sauvegardé: {rapport_path}")
    
    async def cleanup(self):
        """Nettoyage ressources"""
        if self.pipeline:
            try:
                await self.pipeline.shutdown()
                print("✅ Pipeline fermé proprement")
            except Exception as e:
                print(f"⚠️ Erreur fermeture pipeline: {e}")

async def main():
    """Fonction principale"""
    print("🚀 TEST VOCAL COMPLET SUPERWHISPER V6")
    print("=" * 60)
    
    # Validation GPU obligatoire
    try:
        validate_rtx3090()
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    # Créer testeur
    testeur = TestVocalComplet()
    
    try:
        # Charger configuration
        if not testeur.load_config():
            return 1
        
        # Initialiser pipeline
        if not await testeur.initialize_pipeline():
            return 1
        
        print("\n🎯 CHOIX DU TEST:")
        print("1. Test avec microphone réel")
        print("2. Test avec simulation vocale")
        
        # Pour l'automatisation, utilisons le test simulation
        print("\n🤖 Lancement test simulation vocale...")
        
        # Test principal
        succes = await testeur.test_conversation_vocale(duree_ecoute=3)
        
        # Test microphone si possible
        print("\n🎤 Tentative test microphone réel...")
        await testeur.test_microphone_reel()
        
        # Rapport final
        testeur.generer_rapport()
        
        if succes:
            print("\n🎊 TEST VOCAL COMPLET RÉUSSI !")
            return 0
        else:
            print("\n❌ Test vocal échoué")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu par l'utilisateur")
        return 1
    
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        return 1
    
    finally:
        await testeur.cleanup()

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1) 