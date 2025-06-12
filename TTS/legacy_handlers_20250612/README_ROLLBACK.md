# Archive Handlers TTS - 12 juin 2025

## Contexte
Consolidation 15→4 handlers suite Phase 2 Enterprise.
Handlers archivés car non-fonctionnels/redondants selon analyse SuperWhisper V6.

## Handlers Archivés (13 fichiers)
- tts_handler_piper_native.py (défaillant - ne fonctionne pas)
- tts_handler_piper_rtx3090.py (défaillant - erreurs GPU)
- tts_handler_piper_simple.py (non testé)
- tts_handler_piper_french.py (non testé)
- tts_handler_piper_original.py (legacy)
- tts_handler_piper_direct.py (legacy)
- tts_handler_piper_espeak.py (legacy)
- tts_handler_piper_fixed.py (legacy)
- tts_handler_piper_cli.py (legacy - remplacé par version unifiée)
- tts_handler_piper.py (legacy)
- tts_handler_coqui.py (alternatif - non maintenu)
- tts_handler_mvp.py (basique - dépassé)
- tts_handler_fallback.py (interface manquante)

## Architecture Nouvelle (UnifiedTTSManager)
- PiperNativeHandler (GPU RTX 3090 <120ms)
- PiperCliHandler (CPU fallback <1000ms)
- SapiFrenchHandler (Windows SAPI <2000ms)
- SilentEmergencyHandler (Silence ultime <5ms)

## Rollback Complet
```bash
# Restauration handlers
mv TTS/legacy_handlers_20250612/*.py TTS/
rm -rf TTS/legacy_handlers_20250612/

# Restauration Git
git checkout pre-tts-enterprise-consolidation
git branch -D feature/tts-enterprise-consolidation
```

## Rollback Partiel
```bash
# Restauration handler spécifique
cp TTS/legacy_handlers_20250612/tts_handler_X.py TTS/
```

## Contact
SuperWhisper V6 Core Team
Date: 12 juin 2025
Tag de sauvegarde: pre-tts-enterprise-consolidation 