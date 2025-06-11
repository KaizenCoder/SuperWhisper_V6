# TTS Package
"""
Module TTS pour LUXA v1.1
Gestion de la synthèse vocale avec différents engines
"""

# Imports des handlers disponibles
try:
    from .tts_handler_piper_fixed import TTSHandlerPiperFixed
except ImportError:
    pass

try:
    from .tts_handler_piper_rtx3090 import TTSHandlerPiperRTX3090
except ImportError:
    pass

try:
    from .tts_handler_coqui import TTSHandlerCoqui
except ImportError:
    pass

__version__ = "1.0.0"
__author__ = "LUXA Team" 