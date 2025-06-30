#!/usr/bin/env python3
"""
Test de validation des 4 handlers TTS candidats
🎯 Objectif: Valider fonctionnalité avant consolidation 15→4
"""

import os
import sys
import time
import asyncio
import importlib.util
from pathlib import Path

# 🚨 CONFIGURATION GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090 EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

class TTSHandlerTester:
    def __init__(self):
        self.test_text = "Bonjour, ceci est un test de validation du handler TTS"
        self.handlers_to_test = {
            'piper_rtx3090': 'TTS/tts_handler_piper_rtx3090.py',
            'piper_native': 'TTS/tts_handler_piper_native.py', 
            'sapi_french': 'TTS/tts_handler_sapi_french.py',
            'fallback': 'TTS/tts_handler_fallback.py'
        }
        self.results = {}
    
    def import_handler(self, handler_path):
        """Import dynamique d'un handler TTS"""
        try:
            if not os.path.exists(handler_path):
                return None, f"Fichier introuvable: {handler_path}"
            
            spec = importlib.util.spec_from_file_location("handler", handler_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Recherche classe TTS
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    'TTS' in attr_name.upper() and 
                    'Handler' in attr_name):
                    return attr, None
            
            return None, "Aucune classe TTSHandler trouvée"
            
        except Exception as e:
            return None, f"Erreur import: {str(e)}"
    
    async def test_handler(self, handler_name, handler_path):
        """Test individuel d'un handler TTS"""
        print(f"\n🔍 Test Handler: {handler_name}")
        print(f"📁 Fichier: {handler_path}")
        
        result = {
            'handler_name': handler_name,
            'file_exists': False,
            'import_success': False,
            'class_found': False,
            'init_success': False,
            'synthesis_success': False,
            'latency_ms': None,
            'audio_generated': False,
            'error_details': []
        }
        
        # 1. Vérification fichier
        if os.path.exists(handler_path):
            result['file_exists'] = True
            print("✅ Fichier existe")
        else:
            result['error_details'].append(f"Fichier manquant: {handler_path}")
            print("❌ Fichier manquant")
            return result
        
        # 2. Test import
        handler_class, import_error = self.import_handler(handler_path)
        if handler_class:
            result['import_success'] = True
            result['class_found'] = True
            print(f"✅ Import réussi: {handler_class.__name__}")
        else:
            result['error_details'].append(import_error)
            print(f"❌ Import échoué: {import_error}")
            return result
        
        # 3. Test initialisation
        try:
            # Configuration basique pour test
            config = {
                'model_path': 'models/tts/',
                'voice': 'fr',
                'speaker_id': 0
            }
            
            handler_instance = handler_class(config)
            result['init_success'] = True
            print("✅ Initialisation réussie")
            
        except Exception as e:
            result['error_details'].append(f"Erreur init: {str(e)}")
            print(f"❌ Initialisation échouée: {str(e)}")
            return result
        
        # 4. Test synthèse
        try:
            start_time = time.time()
            
            # Test méthode synthèse (peut varier selon handler)
            if hasattr(handler_instance, 'synthesize'):
                audio_output = await handler_instance.synthesize(self.test_text)
            elif hasattr(handler_instance, 'generate_audio'):
                audio_output = await handler_instance.generate_audio(self.test_text)
            elif hasattr(handler_instance, 'text_to_speech'):
                audio_output = await handler_instance.text_to_speech(self.test_text)
            else:
                # Fallback pour handlers différents
                methods = [m for m in dir(handler_instance) if 'synth' in m.lower() or 'generate' in m.lower()]
                if methods:
                    method = getattr(handler_instance, methods[0])
                    audio_output = await method(self.test_text)
                else:
                    raise Exception("Aucune méthode de synthèse trouvée")
            
            latency = (time.time() - start_time) * 1000
            result['latency_ms'] = round(latency, 2)
            
            if audio_output and len(audio_output) > 0:
                result['synthesis_success'] = True
                result['audio_generated'] = True
                print(f"✅ Synthèse réussie - Latence: {latency:.1f}ms")
                print(f"🔊 Audio généré: {len(audio_output)} bytes")
            else:
                result['error_details'].append("Audio vide ou null")
                print("❌ Audio vide généré")
                
        except Exception as e:
            result['error_details'].append(f"Erreur synthèse: {str(e)}")
            print(f"❌ Synthèse échouée: {str(e)}")
        
        return result
    
    async def run_all_tests(self):
        """Exécution de tous les tests handlers"""
        print("🧪 VALIDATION 4 HANDLERS TTS CANDIDATS")
        print("=" * 50)
        
        for handler_name, handler_path in self.handlers_to_test.items():
            result = await self.test_handler(handler_name, handler_path)
            self.results[handler_name] = result
        
        # Rapport final
        self.generate_report()
    
    def generate_report(self):
        """Génération rapport de test"""
        print("\n📊 RAPPORT DE VALIDATION")
        print("=" * 50)
        
        working_handlers = []
        broken_handlers = []
        
        for handler_name, result in self.results.items():
            if result['synthesis_success']:
                working_handlers.append(handler_name)
                status = "✅ FONCTIONNEL"
                latency = f"{result['latency_ms']:.1f}ms" if result['latency_ms'] else "N/A"
            else:
                broken_handlers.append(handler_name)
                status = "❌ DÉFAILLANT"
                latency = "N/A"
            
            print(f"\n🎯 {handler_name.upper()}")
            print(f"   Status: {status}")
            print(f"   Latence: {latency}")
            print(f"   Fichier: {'✅' if result['file_exists'] else '❌'}")
            print(f"   Import: {'✅' if result['import_success'] else '❌'}")
            print(f"   Synthèse: {'✅' if result['synthesis_success'] else '❌'}")
            
            if result['error_details']:
                print(f"   Erreurs: {'; '.join(result['error_details'])}")
        
        print(f"\n🎯 RÉSUMÉ:")
        print(f"   Handlers fonctionnels: {len(working_handlers)}/4")
        print(f"   Handlers défaillants: {len(broken_handlers)}/4")
        
        if working_handlers:
            print(f"\n✅ HANDLERS À CONSERVER:")
            for handler in working_handlers:
                latency = self.results[handler]['latency_ms']
                print(f"   - {handler}: {latency:.1f}ms" if latency else f"   - {handler}")
        
        if broken_handlers:
            print(f"\n❌ HANDLERS À RÉPARER/REMPLACER:")
            for handler in broken_handlers:
                errors = self.results[handler]['error_details']
                print(f"   - {handler}: {errors[0] if errors else 'Erreur inconnue'}")
        
        # Recommandations
        print(f"\n🎯 RECOMMANDATIONS:")
        if len(working_handlers) >= 2:
            print("   ✅ Consolidation possible avec handlers fonctionnels")
        else:
            print("   ⚠️  Nécessité de réparer handlers avant consolidation")
        
        return working_handlers, broken_handlers

async def main():
    """Fonction principale de test"""
    tester = TTSHandlerTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 