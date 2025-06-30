"""
PIPELINE.tests - Tests Unitaires Pipeline SuperWhisper V6
=========================================================
Suite de tests unitaires pour validation pipeline complet

Tests Modules:
- test_pipeline_orchestrator.py : Tests orchestrateur principal
- test_stt_integration.py : Tests intégration STT
- test_llm_integration.py : Tests intégration LLM  
- test_tts_integration.py : Tests intégration TTS
- test_end_to_end.py : Tests pipeline complet E2E
- test_performance.py : Tests performance < 1.2s

Coverage Target: >90% orchestrateur pipeline

Usage:
    pytest PIPELINE/tests/ -v --cov=PIPELINE
"""

__version__ = "1.1.0" 