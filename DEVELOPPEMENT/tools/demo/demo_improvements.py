#!/usr/bin/env python3
"""
Script de Validation Complète - Luxa SuperWhisper V6
===================================================

Démonstrateur des améliorations de sécurité, robustesse et performance.
Ce script illustre toutes les corrections apportées suite au peer review.

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
import numpy as np
import time
import json
import logging
from pathlib import Path
import sys

# Imports des modules améliorés
sys.path.append(str(Path(__file__).parent))

from config.security_config import SecurityConfig
from utils.error_handler import RobustErrorHandler
from Orchestrator.master_handler_robust import RobustMasterHandler

# Configuration logging coloré
class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour démonstration"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Vert
        'WARNING': '\033[33m',  # Jaune
        'ERROR': '\033[31m',    # Rouge
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Appliquer formatter coloré
for handler in logging.root.handlers:
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)

class LuxaSecurityDemo:
    """Démonstration des améliorations de sécurité"""
    
    def __init__(self):
        self.security_config = SecurityConfig(config_dir="demo_config")
        
    def demo_api_key_management(self):
        """Démontre la gestion sécurisée des clés API"""
        
        print("\n🔐 === DÉMONSTRATION SÉCURITÉ - GESTION CLÉS API ===")
        
        # 1. Génération clé API sécurisée
        print("\n1️⃣ Génération clé API sécurisée:")
        api_key = self.security_config.generate_api_key("demo_user")
        print(f"   ✅ Clé générée: {api_key[:20]}... (truncated)")
        print(f"   🔒 Hash stocké de façon sécurisée")
        
        # 2. Validation clé API
        print("\n2️⃣ Validation clé API:")
        user = self.security_config.validate_api_key(api_key)
        print(f"   ✅ Validation réussie pour: {user}")
        
        # Test clé invalide
        invalid_result = self.security_config.validate_api_key("clé_invalide")
        print(f"   ❌ Clé invalide rejetée: {invalid_result}")
        
        # 3. Génération et validation JWT
        print("\n3️⃣ Gestion JWT:")
        token = self.security_config.generate_jwt_token({
            "username": user,
            "permissions": ["audio_processing", "model_access"],
            "role": "user"
        })
        print(f"   ✅ Token JWT généré (longueur: {len(token)} chars)")
        
        # Validation JWT
        decoded = self.security_config.validate_jwt_token(token)
        print(f"   ✅ Token validé: {decoded['username']} - {decoded['permissions']}")
        
        return api_key, token
    
    def demo_input_validation(self):
        """Démontre la validation sécurisée des entrées"""
        
        print("\n🛡️ === DÉMONSTRATION VALIDATION ENTRÉES ===")
        
        # 1. Audio valide
        print("\n1️⃣ Validation audio valide:")
        valid_audio = self._create_wav_audio(16000)  # 1 seconde
        validation = self.security_config.validate_audio_input(valid_audio, "test.wav")
        print(f"   ✅ Audio valide: {validation['valid']}")
        print(f"   📊 Taille: {len(valid_audio)/1024:.1f} KB")
        
        # 2. Audio trop volumineux
        print("\n2️⃣ Rejet audio trop volumineux:")
        large_audio = b"x" * (15 * 1024 * 1024)  # 15MB
        validation = self.security_config.validate_audio_input(large_audio, "large.wav")
        print(f"   ❌ Audio rejeté: {validation['valid']}")
        print(f"   🚫 Erreur: {validation['errors'][0]}")
        
        # 3. Format non supporté
        print("\n3️⃣ Rejet format non supporté:")
        validation = self.security_config.validate_audio_input(b"test", "malware.exe")
        print(f"   ❌ Format rejeté: {validation['valid']}")
        print(f"   🚫 Erreur: {validation['errors'][0]}")
        
        # 4. Détection contenu suspect
        print("\n4️⃣ Détection contenu suspect:")
        suspicious_data = b"MZ\x90\x00" + b"fake_audio_data" * 100
        validation = self.security_config.validate_audio_input(suspicious_data, "suspicious.wav")
        print(f"   ❌ Contenu suspect détecté: {not validation['valid']}")
        
        # 5. Nettoyage texte
        print("\n5️⃣ Nettoyage entrées texte:")
        dirty_text = "Hello\x00\x07world\x1f avec caractères\x0c de contrôle"
        clean_text = self.security_config.sanitize_text_input(dirty_text)
        print(f"   📝 Texte original: {repr(dirty_text)}")
        print(f"   ✨ Texte nettoyé: {repr(clean_text)}")
    
    def _create_wav_audio(self, samples: int) -> bytes:
        """Crée un fichier WAV valide pour tests"""
        # Header WAV minimal
        header = (
            b'RIFF' + (samples * 2 + 36).to_bytes(4, 'little') +
            b'WAVEfmt ' + (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') + (1).to_bytes(2, 'little') +
            (16000).to_bytes(4, 'little') + (32000).to_bytes(4, 'little') +
            (2).to_bytes(2, 'little') + (16).to_bytes(2, 'little') +
            b'data' + (samples * 2).to_bytes(4, 'little')
        )
        
        # Données audio (silence)
        audio_data = np.zeros(samples, dtype=np.int16).tobytes()
        
        return header + audio_data

class LuxaRobustnessDemo:
    """Démonstration des améliorations de robustesse"""
    
    def __init__(self):
        self.error_handler = RobustErrorHandler()
        
    async def demo_circuit_breaker(self):
        """Démontre le fonctionnement des circuit breakers"""
        
        print("\n⚡ === DÉMONSTRATION CIRCUIT BREAKERS ===")
        
        # Enregistrer un composant de test
        self.error_handler.register_component("demo_service", failure_threshold=3, max_retries=2)
        
        # Simuler service défaillant
        failure_count = 0
        
        async def unreliable_service():
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 5:  # 5 premières tentatives échouent
                raise Exception(f"Service failure #{failure_count}")
            
            return f"Success after {failure_count} attempts"
        
        print("\n1️⃣ Test avec service défaillant:")
        
        # Première série d'appels - doit ouvrir le circuit
        for i in range(6):
            try:
                result = await self.error_handler.execute_safe("demo_service", unreliable_service)
                print(f"   ✅ Tentative {i+1}: {result}")
            except Exception as e:
                print(f"   ❌ Tentative {i+1}: {str(e)}")
            
            # Afficher état du circuit
            circuit = self.error_handler.circuit_breakers["demo_service"]
            print(f"      🔄 État circuit: {circuit.state.value} (erreurs: {circuit.metrics.consecutive_errors})")
            
            if circuit.state.value == "open":
                print(f"      🚨 Circuit ouvert après {circuit.metrics.consecutive_errors} erreurs!")
                break
            
            await asyncio.sleep(0.1)
        
        print("\n2️⃣ État du circuit breaker:")
        status = self.error_handler.get_health_status()
        print(f"   📊 Composants sains: {status['healthy_components']}/{status['total_components']}")
        print(f"   📈 Taux d'erreur global: {status['global_metrics']['error_rate']:.2%}")
    
    async def demo_retry_mechanism(self):
        """Démontre le mécanisme de retry"""
        
        print("\n🔄 === DÉMONSTRATION RETRY ===")
        
        # Service qui réussit au 3ème essai
        attempt_count = 0
        
        async def flaky_service():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:
                raise Exception(f"Temporary failure (attempt {attempt_count})")
            
            return f"Success on attempt {attempt_count}!"
        
        print("\n1️⃣ Service avec échecs temporaires:")
        
        try:
            result = await self.error_handler.execute_safe("demo_service", flaky_service)
            print(f"   ✅ Résultat final: {result}")
            print(f"   🔄 Nombre total d'essais: {attempt_count}")
        except Exception as e:
            print(f"   ❌ Échec définitif: {e}")

class LuxaPerformanceDemo:
    """Démonstration des améliorations de performance"""
    
    async def demo_pipeline_performance(self):
        """Démontre les performances du pipeline amélioré"""
        
        print("\n🚀 === DÉMONSTRATION PERFORMANCE ===")
        
        # Initialiser le handler robuste
        handler = RobustMasterHandler()
        await handler.initialize()
        
        print("\n1️⃣ Test latence avec différents types d'audio:")
        
        audio_types = {
            "silence": np.zeros(16000, dtype=np.float32),
            "speech_sim": self._generate_speech_signal(16000),
            "noise": np.random.normal(0, 0.1, 16000).astype(np.float32)
        }
        
        for audio_type, audio_data in audio_types.items():
            latencies = []
            
            # 5 mesures par type
            for i in range(5):
                start_time = time.perf_counter()
                
                result = await handler.process_audio_secure(
                    audio_chunk=audio_data,
                    jwt_token=handler.security_config.generate_jwt_token({"username": "perf_demo"})
                )
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print(f"   📊 {audio_type:12s}: μ={avg_latency:5.1f}ms min={min_latency:5.1f}ms max={max_latency:5.1f}ms")
            
            # Vérifier objectifs
            if audio_type == "silence":
                assert avg_latency < 100, f"Silence trop lent: {avg_latency:.1f}ms"
            else:
                assert avg_latency < 3000, f"{audio_type} trop lent: {avg_latency:.1f}ms"
        
        print("\n2️⃣ Test charge concurrente:")
        
        async def concurrent_request(req_id: int):
            """Requête concurrente"""
            audio = self._generate_speech_signal(8000)  # Audio court
            
            start = time.perf_counter()
            result = await handler.process_audio_secure(
                audio_chunk=audio,
                jwt_token=handler.security_config.generate_jwt_token({"username": f"concurrent_{req_id}"})
            )
            end = time.perf_counter()
            
            return {
                "req_id": req_id,
                "latency_ms": (end - start) * 1000,
                "success": result["success"]
            }
        
        # Test avec 5 requêtes concurrentes
        concurrent_tasks = [concurrent_request(i) for i in range(5)]
        results = await asyncio.gather(*concurrent_tasks)
        
        successful = [r for r in results if r["success"]]
        success_rate = len(successful) / len(results)
        
        if successful:
            avg_concurrent_latency = sum(r["latency_ms"] for r in successful) / len(successful)
            print(f"   ⚡ Requêtes concurrentes: {len(successful)}/{len(results)} réussies")
            print(f"   📊 Latence moyenne: {avg_concurrent_latency:.1f}ms")
            print(f"   🎯 Taux de succès: {success_rate:.1%}")
        
        print("\n3️⃣ État de santé système:")
        health = handler.get_health_status()
        print(f"   🏥 Statut: {health['status']}")
        print(f"   📈 Requêtes traitées: {health['performance']['requests_processed']}")
        print(f"   ⚡ Latence moyenne: {health['performance']['avg_latency_ms']:.1f}ms")
        print(f"   ✅ Taux de succès: {health['performance']['success_rate']:.1%}")
    
    def _generate_speech_signal(self, length: int) -> np.ndarray:
        """Génère un signal similaire à la parole"""
        t = np.linspace(0, length/16000, length)
        
        # Formants typiques
        f1 = 800 + 200 * np.sin(2 * np.pi * 3 * t)
        f2 = 1200 + 300 * np.sin(2 * np.pi * 2 * t)
        
        signal = (
            0.4 * np.sin(2 * np.pi * f1 * t) +
            0.3 * np.sin(2 * np.pi * f2 * t)
        )
        
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 1.5 * t))
        
        return (signal * envelope * 0.7).astype(np.float32)

class LuxaIntegrationDemo:
    """Démonstration d'intégration complète"""
    
    async def demo_complete_workflow(self):
        """Démontre le workflow complet sécurisé"""
        
        print("\n🎯 === DÉMONSTRATION WORKFLOW COMPLET ===")
        
        # 1. Configuration sécurité
        print("\n1️⃣ Configuration sécurité:")
        security = SecurityConfig(config_dir="integration_demo")
        api_key = security.generate_api_key("integration_user")
        print(f"   🔑 Clé API générée pour integration_user")
        
        # 2. Initialisation handler robuste
        print("\n2️⃣ Initialisation handler robuste:")
        handler = RobustMasterHandler()
        await handler.initialize()
        print(f"   ✅ Handler initialisé avec protection complète")
        
        # 3. Validation et traitement audio sécurisé
        print("\n3️⃣ Traitement audio sécurisé:")
        
        # Créer audio de test
        audio_data = np.random.normal(0, 0.1, 16000).astype(np.float32)
        
        # Traitement avec authentification
        result = await handler.process_audio_secure(
            audio_chunk=audio_data,
            api_key=api_key,
            filename="integration_test.wav"
        )
        
        print(f"   ✅ Traitement réussi: {result['success']}")
        print(f"   🔐 Utilisateur authentifié: {result['security']['user']}")
        print(f"   🛡️ Méthode auth: {result['security']['auth_method']}")
        print(f"   📊 Latence: {result['latency_ms']:.1f}ms")
        print(f"   🎤 Composants utilisés: {list(result['components_used'].keys())}")
        
        # 4. Métriques et monitoring
        print("\n4️⃣ Métriques et monitoring:")
        health = handler.get_health_status()
        
        print(f"   📊 Requêtes totales: {health['performance']['requests_processed']}")
        print(f"   ⚡ Latence moyenne: {health['performance']['avg_latency_ms']:.1f}ms")
        print(f"   🔒 Circuits breakers: {len(health['security']['circuit_breakers']['components'])} actifs")
        
        # 5. Export métriques Prometheus
        print("\n5️⃣ Export métriques Prometheus:")
        prometheus_metrics = handler.error_handler.export_metrics_prometheus()
        metrics_lines = prometheus_metrics.split('\n')[:5]  # Premières lignes
        for line in metrics_lines:
            if line.strip():
                print(f"   📈 {line}")
        print(f"   ... (+{len(prometheus_metrics.split()) - 5} métriques)")

async def main():
    """Fonction principale de démonstration"""
    
    print("🎨" + "="*70)
    print("🎨  LUXA SUPERWHISPER V6 - DÉMONSTRATION AMÉLIORATIONS")
    print("🎨  Suite aux recommandations du Peer Review")
    print("🎨" + "="*70)
    
    try:
        # 1. Démonstration sécurité
        security_demo = LuxaSecurityDemo()
        api_key, jwt_token = security_demo.demo_api_key_management()
        security_demo.demo_input_validation()
        
        # 2. Démonstration robustesse
        robustness_demo = LuxaRobustnessDemo()
        await robustness_demo.demo_circuit_breaker()
        await robustness_demo.demo_retry_mechanism()
        
        # 3. Démonstration performance
        performance_demo = LuxaPerformanceDemo()
        await performance_demo.demo_pipeline_performance()
        
        # 4. Démonstration intégration
        integration_demo = LuxaIntegrationDemo()
        await integration_demo.demo_complete_workflow()
        
        print("\n🎉" + "="*70)
        print("🎉  DÉMONSTRATION TERMINÉE AVEC SUCCÈS!")
        print("🎉  Toutes les améliorations sont fonctionnelles:")
        print("🎉  ✅ Sécurité renforcée (auth, validation)")
        print("🎉  ✅ Robustesse améliorée (circuit breakers, retry)")
        print("🎉  ✅ Performance optimisée (latence, débit)")
        print("🎉  ✅ Monitoring complet (métriques, santé)")
        print("🎉" + "="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur durant la démonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Nettoyer les dossiers de test précédents
    import shutil
    for demo_dir in ["demo_config", "integration_demo"]:
        if Path(demo_dir).exists():
            shutil.rmtree(demo_dir)
    
    # Lancer la démonstration
    asyncio.run(main())
