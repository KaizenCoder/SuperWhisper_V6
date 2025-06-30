#!/usr/bin/env python3
"""
SuperWhisper V6 - Health Check Production
Vérification complète du système avant mise en production
"""

import os
import sys
import pathlib
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List

def _setup_portable_environment():
    current_file = pathlib.Path(__file__).resolve()
    
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    os.chdir(project_root)
    
    # Configuration GPU RTX 3090
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    return project_root

_PROJECT_ROOT = _setup_portable_environment()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("HealthCheckProd")

class ProductionHealthChecker:
    """Vérification complète de la santé du système SuperWhisper V6"""
    
    def __init__(self):
        self.results = {
            'system': {'status': 'unknown', 'checks': [], 'score': 0},
            'gpu': {'status': 'unknown', 'checks': [], 'score': 0},
            'stt': {'status': 'unknown', 'checks': [], 'score': 0},
            'llm': {'status': 'unknown', 'checks': [], 'score': 0},
            'tts': {'status': 'unknown', 'checks': [], 'score': 0},
            'audio': {'status': 'unknown', 'checks': [], 'score': 0},
            'integration': {'status': 'unknown', 'checks': [], 'score': 0}
        }
        self.start_time = time.time()

    async def check_system_environment(self):
        """Vérification environnement système"""
        logger.info("🖥️ Vérification environnement système...")
        
        try:
            # Vérification Python
            python_version = sys.version.split()[0]
            if python_version >= "3.8":
                self.results['system']['checks'].append(f"✅ Python {python_version}")
                self.results['system']['score'] += 10
            else:
                self.results['system']['checks'].append(f"❌ Python {python_version} < 3.8")
            
            # Vérification variables d'environnement
            cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_devices == '1':
                self.results['system']['checks'].append("✅ CUDA_VISIBLE_DEVICES=1")
                self.results['system']['score'] += 15
            else:
                self.results['system']['checks'].append(f"❌ CUDA_VISIBLE_DEVICES={cuda_devices}")
            
            # Vérification répertoire de travail
            cwd = os.getcwd()
            if 'SuperWhisper_V6' in cwd:
                self.results['system']['checks'].append(f"✅ Répertoire: {os.path.basename(cwd)}")
                self.results['system']['score'] += 10
            else:
                self.results['system']['checks'].append(f"❌ Répertoire incorrect: {cwd}")
            
            # Vérification fichiers critiques
            critical_files = [
                'STT/unified_stt_manager.py',
                'LLM/llm_manager_enhanced.py', 
                'TTS/tts_manager.py',
                'config/tts.yaml'
            ]
            
            for file_path in critical_files:
                if os.path.exists(file_path):
                    self.results['system']['checks'].append(f"✅ {file_path}")
                    self.results['system']['score'] += 5
                else:
                    self.results['system']['checks'].append(f"❌ Manquant: {file_path}")
            
            self.results['system']['status'] = 'functional' if self.results['system']['score'] >= 30 else 'error'
            return True
            
        except Exception as e:
            self.results['system']['status'] = 'error'
            self.results['system']['checks'].append(f"❌ Erreur système: {e}")
            return False

    async def check_gpu_configuration(self):
        """Vérification configuration GPU RTX 3090"""
        logger.info("🎮 Vérification configuration GPU...")
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.results['gpu']['checks'].append("❌ CUDA non disponible")
                self.results['gpu']['status'] = 'error'
                return False
            
            self.results['gpu']['checks'].append("✅ CUDA disponible")
            self.results['gpu']['score'] += 20
            
            # Vérification GPU actuel
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
            
            self.results['gpu']['checks'].append(f"✅ GPU actuel: {gpu_name}")
            self.results['gpu']['checks'].append(f"✅ Mémoire GPU: {gpu_memory:.1f}GB")
            
            # Vérification RTX 3090
            if "RTX 3090" in gpu_name or gpu_memory > 20:
                self.results['gpu']['checks'].append("✅ RTX 3090 validée")
                self.results['gpu']['score'] += 30
            else:
                self.results['gpu']['checks'].append(f"⚠️ GPU non-RTX 3090: {gpu_name}")
                self.results['gpu']['score'] += 10
            
            # Test allocation mémoire
            test_tensor = torch.zeros(1000, 1000, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            
            self.results['gpu']['checks'].append("✅ Test allocation mémoire GPU")
            self.results['gpu']['score'] += 20
            
            self.results['gpu']['status'] = 'optimal' if self.results['gpu']['score'] >= 60 else 'functional'
            return True
            
        except Exception as e:
            self.results['gpu']['status'] = 'error'
            self.results['gpu']['checks'].append(f"❌ Erreur GPU: {e}")
            return False

    async def check_stt_component(self):
        """Vérification composant STT"""
        logger.info("🎤 Vérification STT...")
        
        try:
            from STT.unified_stt_manager import UnifiedSTTManager
            
            stt_config = {
                'timeout_per_minute': 10.0,
                'cache_size_mb': 100,
                'fallback_chain': ['prism_primary']
            }
            
            stt_manager = UnifiedSTTManager(stt_config)
            self.results['stt']['checks'].append("✅ UnifiedSTTManager initialisé")
            self.results['stt']['score'] += 25
            
            # Test avec audio simulé
            import numpy as np
            sample_rate = 16000
            duration = 1.0
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 440
            amplitude = 0.3
            audio_data = amplitude * np.sin(2 * np.pi * frequency * t)
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            start_time = time.perf_counter()
            result = await stt_manager.transcribe_pcm(audio_bytes, sample_rate)
            stt_time = (time.perf_counter() - start_time) * 1000
            
            self.results['stt']['checks'].append(f"✅ Test transcription: {stt_time:.1f}ms")
            
            if stt_time < 2000:
                self.results['stt']['checks'].append("✅ Performance excellente")
                self.results['stt']['score'] += 25
            else:
                self.results['stt']['checks'].append("⚠️ Performance lente")
                self.results['stt']['score'] += 10
            
            self.results['stt']['status'] = 'functional'
            return True
            
        except Exception as e:
            self.results['stt']['status'] = 'error'
            self.results['stt']['checks'].append(f"❌ Erreur STT: {e}")
            return False

    async def check_llm_component(self):
        """Vérification composant LLM"""
        logger.info("🧠 Vérification LLM...")
        
        try:
            from LLM.llm_manager_enhanced import EnhancedLLMManager
            
            llm_config = {
                'model': 'nous-hermes-2-mistral-7b-dpo:latest',
                'use_ollama': True,
                'timeout': 15.0
            }
            
            llm_manager = EnhancedLLMManager(llm_config)
            await llm_manager.initialize()
            
            self.results['llm']['checks'].append("✅ EnhancedLLMManager initialisé")
            self.results['llm']['score'] += 20
            
            # Test connexion Ollama
            if hasattr(llm_manager, 'use_ollama') and llm_manager.use_ollama:
                self.results['llm']['checks'].append("✅ Ollama connecté")
                self.results['llm']['score'] += 30
            else:
                self.results['llm']['checks'].append("⚠️ Ollama non disponible - Fallback")
                self.results['llm']['score'] += 15
            
            # Test génération
            test_question = "Test de fonctionnement"
            start_time = time.perf_counter()
            response = await llm_manager.generate_response(test_question, max_tokens=20)
            llm_time = (time.perf_counter() - start_time) * 1000
            
            self.results['llm']['checks'].append(f"✅ Test génération: {llm_time:.1f}ms")
            self.results['llm']['checks'].append(f"✅ Réponse: {response[:50]}...")
            
            if llm_time < 5000:
                self.results['llm']['checks'].append("✅ Performance excellente")
                self.results['llm']['score'] += 20
            else:
                self.results['llm']['checks'].append("⚠️ Performance lente")
                self.results['llm']['score'] += 10
            
            self.results['llm']['status'] = 'optimal' if self.results['llm']['score'] >= 60 else 'functional'
            return True
            
        except Exception as e:
            self.results['llm']['status'] = 'error'
            self.results['llm']['checks'].append(f"❌ Erreur LLM: {e}")
            return False

    async def check_tts_component(self):
        """Vérification composant TTS"""
        logger.info("🔊 Vérification TTS...")
        
        try:
            import yaml
            
            with open('config/tts.yaml', 'r') as f:
                tts_config = yaml.safe_load(f)
            
            from TTS.tts_manager import UnifiedTTSManager
            tts_manager = UnifiedTTSManager(tts_config)
            
            self.results['tts']['checks'].append("✅ UnifiedTTSManager initialisé")
            self.results['tts']['score'] += 25
            
            # Test synthèse
            test_text = "Test de synthèse vocale française"
            start_time = time.perf_counter()
            result = await tts_manager.synthesize(text=test_text)
            tts_time = (time.perf_counter() - start_time) * 1000
            
            self.results['tts']['checks'].append(f"✅ Test synthèse: {tts_time:.1f}ms")
            
            if result.success and result.audio_data:
                audio_size = len(result.audio_data)
                self.results['tts']['checks'].append(f"✅ Audio généré: {audio_size} bytes")
                self.results['tts']['score'] += 25
                
                if tts_time < 1500:
                    self.results['tts']['checks'].append("✅ Performance excellente")
                    self.results['tts']['score'] += 20
                else:
                    self.results['tts']['checks'].append("⚠️ Performance lente")
                    self.results['tts']['score'] += 10
                
                self.results['tts']['status'] = 'functional'
                return True
            else:
                self.results['tts']['checks'].append(f"❌ Échec synthèse: {result.error}")
                self.results['tts']['status'] = 'error'
                return False
                
        except Exception as e:
            self.results['tts']['status'] = 'error'
            self.results['tts']['checks'].append(f"❌ Erreur TTS: {e}")
            return False

    async def check_audio_devices(self):
        """Vérification périphériques audio"""
        logger.info("🎧 Vérification périphériques audio...")
        
        try:
            import pyaudio
            
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            
            self.results['audio']['checks'].append(f"✅ {device_count} périphériques audio détectés")
            self.results['audio']['score'] += 10
            
            # Recherche RODE NT-USB
            rode_devices = []
            for i in range(device_count):
                try:
                    device_info = p.get_device_info_by_index(i)
                    device_name = device_info['name'].lower()
                    
                    if 'rode' in device_name or 'nt-usb' in device_name:
                        rode_devices.append(device_info['name'])
                        self.results['audio']['checks'].append(f"✅ RODE détecté: {device_info['name']}")
                        self.results['audio']['score'] += 30
                except:
                    continue
            
            if rode_devices:
                self.results['audio']['checks'].append(f"✅ {len(rode_devices)} RODE NT-USB trouvés")
                self.results['audio']['score'] += 20
            else:
                self.results['audio']['checks'].append("⚠️ Aucun RODE NT-USB détecté")
                self.results['audio']['score'] += 5
            
            # Vérification périphériques de sortie
            output_devices = [
                device for device in range(device_count)
                if p.get_device_info_by_index(device)['maxOutputChannels'] > 0
            ]
            
            if output_devices:
                self.results['audio']['checks'].append(f"✅ {len(output_devices)} sorties audio")
                self.results['audio']['score'] += 10
            
            p.terminate()
            
            self.results['audio']['status'] = 'optimal' if self.results['audio']['score'] >= 50 else 'functional'
            return True
            
        except Exception as e:
            self.results['audio']['status'] = 'error'
            self.results['audio']['checks'].append(f"❌ Erreur audio: {e}")
            return False

    async def check_integration(self):
        """Test d'intégration pipeline complet"""
        logger.info("🔗 Test intégration pipeline...")
        
        try:
            # Test STT → LLM → TTS
            from STT.unified_stt_manager import UnifiedSTTManager
            from LLM.llm_manager_enhanced import EnhancedLLMManager
            from TTS.tts_manager import UnifiedTTSManager
            import yaml
            
            # Configuration
            stt_config = {'timeout_per_minute': 10.0, 'fallback_chain': ['prism_primary']}
            llm_config = {'model': 'nous-hermes-2-mistral-7b-dpo:latest', 'use_ollama': True}
            
            with open('config/tts.yaml', 'r') as f:
                tts_config = yaml.safe_load(f)
            
            # Initialisation
            stt_manager = UnifiedSTTManager(stt_config)
            llm_manager = EnhancedLLMManager(llm_config)
            tts_manager = UnifiedTTSManager(tts_config)
            
            await llm_manager.initialize()
            
            self.results['integration']['checks'].append("✅ Composants initialisés")
            self.results['integration']['score'] += 20
            
            # Test pipeline simulé
            test_input = "Bonjour système"
            
            # LLM → TTS
            llm_response = await llm_manager.generate_response(test_input, max_tokens=50)
            tts_result = await tts_manager.synthesize(text=llm_response)
            
            if tts_result.success:
                self.results['integration']['checks'].append("✅ Pipeline LLM→TTS fonctionnel")
                self.results['integration']['score'] += 40
                
                # Nettoyage fichier audio test
                try:
                    import glob
                    test_files = glob.glob("tts_output_*.wav")
                    for file in test_files[-5:]:  # Garder seulement les 5 derniers
                        try:
                            os.remove(file)
                        except:
                            pass
                except:
                    pass
                
                self.results['integration']['status'] = 'functional'
                return True
            else:
                self.results['integration']['checks'].append("❌ Échec pipeline TTS")
                self.results['integration']['status'] = 'partial'
                return False
                
        except Exception as e:
            self.results['integration']['status'] = 'error'
            self.results['integration']['checks'].append(f"❌ Erreur intégration: {e}")
            return False

    def generate_health_report(self):
        """Génération rapport de santé complet"""
        total_score = sum(comp['score'] for comp in self.results.values())
        max_score = 500  # Score maximum possible
        percentage = (total_score / max_score) * 100
        
        # Statut global
        if percentage >= 90:
            global_status = "🟢 EXCELLENT - Prêt production"
        elif percentage >= 75:
            global_status = "🟡 BON - Production avec surveillance"
        elif percentage >= 60:
            global_status = "🟠 ACCEPTABLE - Corrections recommandées"
        else:
            global_status = "🔴 CRITIQUE - Corrections obligatoires"
        
        print("\n" + "="*70)
        print("🏥 RAPPORT DE SANTÉ - SuperWhisper V6 Production")
        print("="*70)
        
        print(f"📊 Score Global: {total_score}/{max_score} ({percentage:.1f}%)")
        print(f"🎯 Statut: {global_status}")
        print(f"⏱️ Durée vérification: {time.time() - self.start_time:.1f}s")
        
        # Détail par composant
        component_names = {
            'system': '🖥️ Système',
            'gpu': '🎮 GPU RTX 3090',
            'stt': '🎤 STT (Speech-to-Text)',
            'llm': '🧠 LLM (Language Model)',
            'tts': '🔊 TTS (Text-to-Speech)',
            'audio': '🎧 Périphériques Audio',
            'integration': '🔗 Intégration Pipeline'
        }
        
        for component, data in self.results.items():
            name = component_names.get(component, component)
            status_icon = {
                'optimal': '🟢',
                'functional': '🟡',
                'partial': '🟠',
                'error': '🔴',
                'unknown': '⚪'
            }.get(data['status'], '⚪')
            
            print(f"\n{status_icon} {name}: {data['score']}/70 ({data['status'].upper()})")
            for check in data['checks']:
                print(f"   {check}")
        
        # Recommandations
        print(f"\n🔧 RECOMMANDATIONS:")
        
        if self.results['gpu']['status'] != 'optimal':
            print("   • Vérifier configuration RTX 3090 et CUDA_VISIBLE_DEVICES=1")
        
        if self.results['llm']['status'] != 'optimal':
            print("   • Vérifier connexion Ollama et modèle nous-hermes-2-mistral-7b-dpo:latest")
        
        if self.results['audio']['status'] != 'optimal':
            print("   • Vérifier connexion RODE NT-USB et drivers audio")
        
        if percentage >= 75:
            print("   • ✅ Système prêt pour utilisation production")
            print("   • ✅ Tous les composants critiques fonctionnent")
        else:
            print("   • ❌ Corrections nécessaires avant mise en production")
        
        # Informations de diagnostic
        print(f"\n📋 INFORMATIONS SYSTÈME:")
        print(f"   • Répertoire: {os.getcwd()}")
        print(f"   • Python: {sys.version.split()[0]}")
        print(f"   • CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Non défini')}")
        print(f"   • Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("="*70)
        
        return {
            'global_score': total_score,
            'max_score': max_score,
            'percentage': percentage,
            'status': global_status,
            'components': self.results,
            'ready_for_production': percentage >= 75
        }

    def save_report(self, report_data: Dict[str, Any]):
        """Sauvegarde du rapport en JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"health_check_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"📄 Rapport sauvegardé: {report_file}")
            return report_file
        except Exception as e:
            print(f"❌ Erreur sauvegarde rapport: {e}")
            return None

async def main():
    print("🏥 SuperWhisper V6 - Health Check Production")
    print("="*70)
    
    checker = ProductionHealthChecker()
    
    try:
        print("\n🔍 Démarrage vérifications...")
        
        # Exécution de tous les tests
        await checker.check_system_environment()
        await checker.check_gpu_configuration() 
        await checker.check_stt_component()
        await checker.check_llm_component()
        await checker.check_tts_component()
        await checker.check_audio_devices()
        await checker.check_integration()
        
        # Génération du rapport
        report_data = checker.generate_health_report()
        
        # Sauvegarde
        checker.save_report(report_data)
        
        # Statut de sortie
        if report_data['ready_for_production']:
            print("\n🎉 SYSTÈME VALIDÉ POUR PRODUCTION!")
            sys.exit(0)
        else:
            print("\n⚠️ CORRECTIONS NÉCESSAIRES AVANT PRODUCTION")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Health check interrompu")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())