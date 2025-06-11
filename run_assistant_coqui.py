# run_assistant_coqui.py
import yaml
import os
import sys
import asyncio

# Ajouter le répertoire courant au PYTHONPATH pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from STT.stt_handler import STTHandler
from LLM.llm_manager_enhanced import EnhancedLLMManager
from LUXA_TTS.tts_handler_coqui import TTSHandlerCoqui

async def main():
    """Assistant vocal LUXA MVP P0 avec Coqui-TTS (100% local)."""
    print("🚀 Démarrage de l'assistant vocal LUXA (MVP P0) - Version Coqui-TTS")
    print("🔐 Mode 100% local - Aucune donnée ne quitte votre machine")

    # 1. Charger la configuration
    try:
        with open("Config/mvp_settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ ERREUR: Le fichier 'Config/mvp_settings.yaml' est introuvable.")
        return

    # 2. Initialiser les modules
    try:
        print("🔧 Initialisation des modules...")
        stt_handler = STTHandler(config['stt'])
        llm_handler = EnhancedLLMManager(config['llm'])
        await llm_handler.initialize()
        tts_handler = TTSHandlerCoqui(config['tts'])
        print("✅ Tous les modules sont initialisés!")
        
        # Test rapide TTS
        print("\n🧪 Test rapide du TTS...")
        tts_handler.test_synthesis()
        
    except Exception as e:
        print(f"❌ ERREUR lors de l'initialisation: {e}")
        print(f"   Détails: {str(e)}")
        return

    # 3. Boucle principale de l'assistant
    print("\n🎯 Assistant vocal LUXA prêt!")
    print("🔐 Fonctionnement 100% local garanti")
    print("Appuyez sur Ctrl+C pour arrêter")
    
    try:
        while True:
            print("\n" + "="*50)
            input("Appuyez sur Entrée pour commencer l'écoute...")
            
            # Pipeline STT → LLM → TTS (100% local)
            try:
                # 1. Écouter et transcrire (local)
                transcription = stt_handler.listen_and_transcribe(duration=5)
                
                if transcription.strip():
                    print(f"📝 Transcription: '{transcription}'")
                    
                    # 2. Générer une réponse (local)
                    response = await llm_handler.generate_response(transcription)
                    
                    if response.strip():
                        # 3. Prononcer la réponse (local)
                        tts_handler.speak(response)
                    else:
                        print("⚠️ Le LLM n'a pas généré de réponse.")
                else:
                    print("⚠️ Aucune parole détectée, réessayez.")
                    
            except Exception as e:
                print(f"❌ Erreur dans le pipeline: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n🛑 Arrêt de l'assistant vocal LUXA")
        print("🔐 Aucune donnée n'a quitté votre machine durant cette session")

if __name__ == "__main__":
    asyncio.run(main()) 