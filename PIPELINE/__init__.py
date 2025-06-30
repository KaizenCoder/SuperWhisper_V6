"""
PIPELINE - Module SuperWhisper V6 Pipeline Complet
==================================================
Pipeline voix-à-voix intégré STT → LLM → TTS

Architecture:
- pipeline_orchestrator.py : Orchestrateur principal v1.1
- config/ : Configurations YAML pipeline
- utils/ : Utilitaires et helpers
- scripts/ : Scripts de validation et démo
- tests/ : Suite de tests unitaires

Usage:
    from PIPELINE.pipeline_orchestrator import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator()
    await orchestrator.process_voice_to_voice(audio_data)

Configuration RTX 3090 obligatoire:
    CUDA_VISIBLE_DEVICES='1' (RTX 3090 24GB exclusif)
"""

__version__ = "1.1.0"
__author__ = "SuperWhisper V6 Team"

from .pipeline_orchestrator import PipelineOrchestrator

__all__ = ["PipelineOrchestrator"] 