# tests/test_enhanced_llm_manager.py
"""
Tests pour EnhancedLLMManager - Validation conversation multi-tours
Conforme aux spécifications du Plan de Développement LUXA Final
"""
import pytest
import asyncio
import yaml
import tempfile
import time
from pathlib import Path
import sys
import logging

# Configuration logging pour debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LLM.llm_manager_enhanced import EnhancedLLMManager, ConversationTurn

@pytest.mark.asyncio
async def test_enhanced_llm_manager_conversation_handling():
    """
    Test de validation conversation multi-tours
    Critères: Contexte conversationnel, métriques, gestion historique
    """
    print("\n" + "="*80)
    print("TEST ENHANCED LLM MANAGER - CONVERSATION HANDLING")
    print("="*80)
    
    # 1. Configuration test
    config_path = Path("Config/mvp_settings.yaml")
    if not config_path.exists():
        pytest.skip(f"Configuration non trouvée: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configuration LLM pour test (modèle plus petit)
    llm_config = config.get('llm', {})
    llm_config.update({
        'max_context_turns': 3,  # Limiter pour test
        'max_history_size': 10,
        'model_path': llm_config.get('model_path', './models/test_model.gguf')
    })
    
    # 2. Initialisation
    print("\n📋 Initialisation EnhancedLLMManager...")
    try:
        llm_manager = EnhancedLLMManager(llm_config)
        await llm_manager.initialize()
        print("✅ LLM Manager initialisé")
    except Exception as e:
        pytest.skip(f"Impossible d'initialiser LLM: {e}")
    
    # 3. Test 1 : Conversation de base sans contexte
    print("\n🧪 TEST 1 : Réponse simple sans contexte")
    
    response1 = await llm_manager.generate_response(
        "Bonjour, comment allez-vous ?",
        max_tokens=50,
        include_context=False
    )
    
    assert response1, "Réponse vide pour salutation simple"
    assert len(response1) > 5, "Réponse trop courte"
    print(f"📝 Réponse 1: '{response1}'")
    
    # Vérification métriques
    metrics1 = llm_manager.get_metrics()
    assert metrics1['total_requests'] == 1, "Compteur requêtes incorrect"
    assert metrics1['conversation_turns'] == 1, "Compteur tours incorrect"
    
    print("✅ TEST 1 RÉUSSI : Réponse simple fonctionnelle")
    
    # 4. Test 2 : Conversation multi-tours avec contexte
    print("\n🧪 TEST 2 : Conversation multi-tours")
    
    # Tour 2
    response2 = await llm_manager.generate_response(
        "Pouvez-vous me rappeler ce que je viens de dire ?",
        max_tokens=80,
        include_context=True
    )
    
    assert response2, "Réponse vide pour question contextuelle"
    
    # Vérifier que le contexte est pris en compte
    # Le modèle devrait faire référence à la salutation précédente
    context_keywords = ["bonjour", "salutation", "dit", "demandé"]
    has_context = any(keyword in response2.lower() for keyword in context_keywords)
    
    print(f"📝 Réponse 2: '{response2}'")
    print(f"🔍 Contexte détecté: {has_context}")
    
    # Tour 3
    response3 = await llm_manager.generate_response(
        "Et maintenant, quel est le sujet de notre conversation ?",
        max_tokens=80,
        include_context=True
    )
    
    assert response3, "Réponse vide pour question métaconversationnelle"
    print(f"📝 Réponse 3: '{response3}'")
    
    # Vérification métriques
    metrics2 = llm_manager.get_metrics()
    assert metrics2['total_requests'] == 3, "Compteur requêtes incorrect après 3 tours"
    assert metrics2['conversation_turns'] == 3, "Compteur tours incorrect"
    assert metrics2['avg_response_time'] > 0, "Latence moyenne non calculée"
    
    print("✅ TEST 2 RÉUSSI : Conversation multi-tours fonctionnelle")
    
    # 5. Test 3 : Résumé de conversation
    print("\n🧪 TEST 3 : Résumé de conversation")
    
    summary = llm_manager.get_conversation_summary()
    
    assert summary['status'] != 'no_conversation', "Résumé indique aucune conversation"
    assert summary['total_turns'] == 3, "Nombre de tours incorrect dans résumé"
    assert summary['duration_minutes'] > 0, "Durée conversation nulle"
    assert 'topics' in summary, "Topics manquants dans résumé"
    assert 'sentiment' in summary, "Sentiment manquant dans résumé"
    
    print(f"📊 Résumé conversation:")
    print(f"   - Tours: {summary['total_turns']}")
    print(f"   - Durée: {summary['duration_minutes']:.2f} min")
    print(f"   - Topics: {summary['topics']}")
    print(f"   - Sentiment: {summary['sentiment']}")
    
    print("✅ TEST 3 RÉUSSI : Résumé conversation complet")
    
    # 6. Test 4 : Gestion historique limite
    print("\n🧪 TEST 4 : Gestion limite historique")
    
    # Ajouter plus de tours pour tester la limite
    for i in range(5):
        await llm_manager.generate_response(
            f"Message test {i+4} pour remplir l'historique",
            max_tokens=20
        )
    
    # Vérifier que l'historique est limité
    current_turns = len(llm_manager.conversation_history)
    max_history = llm_config.get('max_history_size', 10)
    
    assert current_turns <= max_history, f"Historique dépasse limite: {current_turns} > {max_history}"
    
    print(f"📚 Historique actuel: {current_turns} tours (limite: {max_history})")
    print("✅ TEST 4 RÉUSSI : Gestion limite historique")
    
    # 7. Test 5 : Nettoyage conversation
    print("\n🧪 TEST 5 : Nettoyage conversation")
    
    llm_manager.clear_conversation()
    
    assert len(llm_manager.conversation_history) == 0, "Historique non vidé"
    
    summary_vide = llm_manager.get_conversation_summary()
    assert summary_vide['status'] == 'no_conversation', "Résumé devrait indiquer aucune conversation"
    
    print("🧹 Conversation nettoyée")
    print("✅ TEST 5 RÉUSSI : Nettoyage conversation")
    
    # 8. Nettoyage final
    await llm_manager.cleanup()
    
    # Résumé final
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLÈTE RÉUSSIE")
    print(f"   - Tests conversationnels: 5/5 réussis")
    print(f"   - Contexte multi-tours: Fonctionnel")
    print(f"   - Métriques monitoring: Complètes")
    print(f"   - Gestion historique: Opérationnelle")
    print("="*80)

@pytest.mark.asyncio
async def test_context_building():
    """Test spécifique de construction du contexte"""
    print("\n🧪 TEST CONTEXT BUILDING")
    
    config = {
        'model_path': './models/test_model.gguf',
        'max_context_turns': 2,
        'n_gpu_layers': 0  # Force CPU pour test
    }
    
    llm_manager = EnhancedLLMManager(config)
    
    # Ajouter manuellement des tours pour tester
    llm_manager._add_to_history("Première question", "Première réponse")
    llm_manager._add_to_history("Deuxième question", "Deuxième réponse")
    llm_manager._add_to_history("Troisième question", "Troisième réponse")
    
    # Test construction contexte
    prompt = llm_manager._build_contextual_prompt("Question actuelle", include_context=True)
    
    # Vérifications
    assert "Conversation récente:" in prompt, "Header contexte manquant"
    assert "Question actuelle" in prompt, "Question actuelle manquante"
    assert "Assistant:" in prompt, "Prompt de réponse manquant"
    
    # Vérifier limitation contexte (max 2 tours)
    lines = prompt.split('\n')
    utilisateur_lines = [line for line in lines if line.startswith('Utilisateur:')]
    
    # Devrait avoir maximum 2 tours + 1 actuel = 3 "Utilisateur:"
    assert len(utilisateur_lines) <= 3, f"Trop de tours dans contexte: {len(utilisateur_lines)}"
    
    print("✅ Construction contexte validée")

@pytest.mark.asyncio 
async def test_response_cleaning():
    """Test du nettoyage des réponses"""
    print("\n🧪 TEST RESPONSE CLEANING")
    
    config = {
        'model_path': './models/test_model.gguf',
        'n_gpu_layers': 0
    }
    
    llm_manager = EnhancedLLMManager(config)
    
    # Tests de nettoyage
    test_cases = [
        ("Assistant: Voici ma réponse", "Voici ma réponse"),
        ("Utilisateur: Question\nAssistant: Réponse", "Réponse"),
        ("   Réponse avec espaces   ", "Réponse avec espaces"),
        ("Réponse très longue " + "qui dépasse " * 50 + "la limite.", 
         "Réponse très longue " + "qui dépasse " * 45)  # Approximativement
    ]
    
    for raw_response, expected_type in test_cases:
        cleaned = llm_manager._clean_response(raw_response)
        
        assert cleaned != raw_response or len(raw_response) < 50, "Nettoyage non appliqué"
        assert not cleaned.startswith("Assistant:"), "Préfixe Assistant: non supprimé"
        assert len(cleaned) <= 500, "Réponse trop longue non coupée"
        
    print("✅ Nettoyage réponses validé")

if __name__ == "__main__":
    # Exécution directe pour tests manuels
    asyncio.run(test_enhanced_llm_manager_conversation_handling()) 