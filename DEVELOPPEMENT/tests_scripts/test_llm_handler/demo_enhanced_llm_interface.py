# tests/demo_enhanced_llm_interface.py
"""
Démonstration de l'interface utilisateur avec EnhancedLLMManager
Validation de l'intégration complète selon PRD v3.1
"""
import asyncio
import yaml
from pathlib import Path
import sys
import time

# Ajout du répertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LLM.llm_manager_enhanced import EnhancedLLMManager

async def demo_conversation_interface():
    """Démonstration interactive de l'interface conversationnelle"""
    print("\n" + "="*80)
    print("🎤 DÉMONSTRATION INTERFACE CONVERSATIONNELLE LUXA")
    print("EnhancedLLMManager - Validation Interface Utilisateur")
    print("="*80)
    
    # 1. Configuration
    config_path = Path("Config/mvp_settings.yaml")
    if not config_path.exists():
        print(f"❌ Configuration non trouvée: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configuration optimisée pour démo
    llm_config = config.get('llm', {})
    llm_config.update({
        'max_context_turns': 5,  # Contexte étendu pour démo
        'max_history_size': 20,
        'model_path': llm_config.get('model_path', './models/llm_model.gguf')
    })
    
    # 2. Initialisation
    print("\n🚀 Initialisation EnhancedLLMManager...")
    try:
        llm_manager = EnhancedLLMManager(llm_config)
        await llm_manager.initialize()
        print("✅ LLM Manager prêt pour conversation")
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        return
    
    # 3. Démonstration conversation interactive
    print("\n💬 Mode Conversation Interactive")
    print("Commandes spéciales:")
    print("  - 'quit' : Quitter")
    print("  - 'status' : Voir statut conversation")
    print("  - 'clear' : Vider historique")
    print("  - 'summary' : Résumé conversation")
    print("-" * 50)
    
    try:
        while True:
            # Interface utilisateur
            try:
                user_input = input("\n🗣️ Vous: ").strip()
                
                if not user_input:
                    continue
                
                # Commandes spéciales
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Au revoir! Merci pour cette conversation.")
                    break
                
                elif user_input.lower() == 'status':
                    status = llm_manager.get_status()
                    print(f"\n📊 Statut LLM Manager:")
                    print(f"   - Modèle chargé: {status['model_loaded']}")
                    print(f"   - Tours conversation: {status['conversation_turns']}")
                    print(f"   - Contexte mémoire: {status['memory_context']} caractères")
                    print(f"   - Requêtes totales: {status['metrics']['total_requests']}")
                    print(f"   - Latence moyenne: {status['metrics']['avg_response_time']:.3f}s")
                    continue
                
                elif user_input.lower() == 'clear':
                    llm_manager.clear_conversation()
                    print("🧹 Historique conversation effacé")
                    continue
                
                elif user_input.lower() == 'summary':
                    summary = llm_manager.get_conversation_summary()
                    if summary['status'] == 'no_conversation':
                        print("📝 Aucune conversation en cours")
                    else:
                        print(f"\n📋 Résumé Conversation:")
                        print(f"   - Tours: {summary['total_turns']}")
                        print(f"   - Durée: {summary['duration_minutes']:.1f} min")
                        print(f"   - Dernière interaction: {summary['last_interaction']:.1f}s")
                        print(f"   - Topics: {', '.join(summary['topics'][:3])}")
                        print(f"   - Sentiment: {summary['sentiment']}")
                    continue
                
                # Génération réponse normale
                print("🧠 LUXA réfléchit...")
                start_time = time.time()
                
                response = await llm_manager.generate_response(
                    user_input,
                    max_tokens=150,
                    temperature=0.7,
                    include_context=True
                )
                
                latency = time.time() - start_time
                
                # Affichage réponse avec métriques
                print(f"🤖 LUXA: {response}")
                print(f"⏱️ Latence: {latency:.2f}s")
                
                # Indicateur contexte
                if len(llm_manager.conversation_history) > 1:
                    print(f"💭 Contexte: {len(llm_manager.conversation_history)} tours")
                
            except KeyboardInterrupt:
                print("\n🛑 Interruption utilisateur...")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                continue
    
    finally:
        # 4. Rapport final et nettoyage
        print("\n" + "="*50)
        print("📊 RAPPORT FINAL CONVERSATION")
        print("="*50)
        
        # Métriques finales
        final_metrics = llm_manager.get_metrics()
        print(f"📈 Métriques Session:")
        print(f"   - Requêtes traitées: {final_metrics['total_requests']}")
        print(f"   - Tours conversation: {final_metrics['conversation_turns']}")
        print(f"   - Tokens générés: {final_metrics['total_tokens_generated']}")
        print(f"   - Resets contexte: {final_metrics['context_resets']}")
        print(f"   - Latence moyenne: {final_metrics['avg_response_time']:.3f}s")
        
        # Résumé final si conversation
        if final_metrics['conversation_turns'] > 0:
            final_summary = llm_manager.get_conversation_summary()
            print(f"\n🎯 Résumé Final:")
            print(f"   - Durée totale: {final_summary['duration_minutes']:.1f} min")
            print(f"   - Topics abordés: {', '.join(final_summary['topics'][:5])}")
            print(f"   - Sentiment global: {final_summary['sentiment']}")
        
        # Nettoyage
        await llm_manager.cleanup()
        print("\n✅ Démonstration terminée - EnhancedLLMManager validé")

async def demo_performance_test():
    """Test de performance de l'interface"""
    print("\n🚀 TEST PERFORMANCE INTERFACE")
    
    config_path = Path("Config/mvp_settings.yaml")
    if not config_path.exists():
        print("Configuration manquante")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    llm_manager = EnhancedLLMManager(config['llm'])
    
    try:
        await llm_manager.initialize()
        
        # Test rapidité réponses
        test_prompts = [
            "Bonjour",
            "Comment allez-vous ?",
            "Quel temps fait-il ?",
            "Racontez-moi une blague",
            "Au revoir"
        ]
        
        print(f"⏱️ Test {len(test_prompts)} requêtes...")
        
        total_start = time.time()
        latencies = []
        
        for i, prompt in enumerate(test_prompts, 1):
            start = time.time()
            response = await llm_manager.generate_response(prompt, max_tokens=50)
            latency = time.time() - start
            latencies.append(latency)
            
            print(f"  {i}. '{prompt}' → {latency:.2f}s")
        
        total_time = time.time() - total_start
        avg_latency = sum(latencies) / len(latencies)
        
        print(f"\n📊 Résultats Performance:")
        print(f"   - Temps total: {total_time:.2f}s")
        print(f"   - Latence moyenne: {avg_latency:.2f}s")
        print(f"   - Latence max: {max(latencies):.2f}s")
        print(f"   - Latence min: {min(latencies):.2f}s")
        
        # Vérification critères PRD v3.1
        target_latency = 0.5  # 500ms objectif
        success_rate = sum(1 for lat in latencies if lat < target_latency) / len(latencies)
        
        print(f"\n🎯 Conformité PRD v3.1:")
        print(f"   - Cible latence: <{target_latency}s")
        print(f"   - Taux succès: {success_rate*100:.1f}%")
        
        if success_rate >= 0.8:
            print("✅ Performance acceptable")
        else:
            print("⚠️ Performance à améliorer")
    
    finally:
        await llm_manager.cleanup()

async def main():
    """Fonction principale de démonstration"""
    print("🎮 DÉMONSTRATION ENHANCED LLM MANAGER")
    print("Choisissez votre mode:")
    print("1. Conversation interactive")
    print("2. Test de performance")
    print("3. Les deux")
    
    choice = input("\nVotre choix (1-3): ").strip()
    
    if choice == "1":
        await demo_conversation_interface()
    elif choice == "2":
        await demo_performance_test()
    elif choice == "3":
        await demo_conversation_interface()
        await demo_performance_test()
    else:
        print("Choix invalide")

if __name__ == "__main__":
    asyncio.run(main()) 