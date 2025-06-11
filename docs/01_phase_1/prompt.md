 PROMPT FINAL : Finalisation, Instrumentation et Validation du MVP P0 de LUXA

## 1. Objectif Global
Ta mission est d'exécuter le plan d'action final pour officiellement clore la Phase 0 du projet Luxa. Cela implique de créer un script de validation, de corriger un bug de test, d'instrumenter le code principal pour mesurer la latence, et de mettre à jour la documentation pour refléter que le MVP est désormais 100% validé.

## 2. Plan d'Action Séquentiel

Exécute les tâches suivantes dans cet ordre précis.

### Tâche 1 : Créer le script de validation PowerShell
Crée le fichier `validate_piper.ps1` à la racine du projet. Ce script validera le moteur TTS en isolation.

**Contenu pour `validate_piper.ps1` :**
```powershell
Write-Host "🧪 Validation de bas niveau pour piper.exe..."
$PiperExecutable = ".\piper\piper.exe"
$ModelPath = ".\models\fr_FR-siwis-medium.onnx"
$OutputFile = "validation_output.wav"
$TestText = "Si vous entendez cette phrase, la validation de base de Piper est réussie."

if (-not (Test-Path $PiperExecutable)) {
    Write-Error "❌ ERREUR: L'exécutable Piper n'a pas été trouvé à l'emplacement '$PiperExecutable'."
    exit 1
}
if (-not (Test-Path $ModelPath)) {
    Write-Error "❌ ERREUR: Le modèle '$ModelPath' n'a pas été trouvé."
    exit 1
}

Write-Host "✅ Prérequis validés."
Write-Host "🔊 Lancement de la synthèse..."
echo $TestText | & $PiperExecutable --model $ModelPath --output_file $OutputFile --speaker 0 --use_gpu
if (Test-Path $OutputFile) {
    Write-Host "✅ Fichier '$OutputFile' généré. Écoutez-le pour valider."
    Invoke-Item $OutputFile
} else {
    Write-Error "❌ La génération a échoué."
}
Tâche 2 : Corriger le script de test test_tts_handler.py
Modifie ce fichier pour qu'il utilise le bon modèle (siwis) dans ses messages d'affichage afin de correspondre à la configuration mvp_settings.yaml.

Fichier à modifier : test_tts_handler.py

Python

# Change this line:
print("🧪 Test du TTSHandler avec le modèle fr_FR-upmc-medium")
# To this:
print("🧪 Test du TTSHandler avec le modèle fr_FR-siwis-medium")
Tâche 3 : Instrumenter la latence dans run_assistant.py
Modifie la boucle principale de run_assistant.py pour chronométrer chaque étape et la durée totale du pipeline.

Fichier à modifier : run_assistant.py

Python

# Add this import at the top
import time

# ... (main function and initializations) ...

# Inside the 'while True:' loop
try:
    total_start_time = time.perf_counter()
    
    # Étape STT
    stt_start_time = time.perf_counter()
    user_text = stt_handler.listen_and_transcribe(duration=7)
    stt_latency = time.perf_counter() - stt_start_time

    if user_text and user_text.strip():
        # Étape LLM
        llm_start_time = time.perf_counter()
        llm_response = llm_handler.get_response(user_text)
        llm_latency = time.perf_counter() - llm_start_time

        # Étape TTS
        tts_start_time = time.perf_counter()
        if llm_response and llm_response.strip():
            tts_handler.speak(llm_response)
        tts_latency = time.perf_counter() - tts_start_time
        
        total_latency = time.perf_counter() - total_start_time
        
        print("\n--- 📊 Rapport de Latence ---")
        print(f"  - STT: {stt_latency:.3f}s")
        print(f"  - LLM: {llm_latency:.3f}s")
        print(f"  - TTS: {tts_latency:.3f}s")
        print(f"  - TOTAL: {total_latency:.3f}s")
        print("----------------------------\n")

    else:
        print("Aucun texte intelligible n'a été transcrit, nouvelle écoute...")

except KeyboardInterrupt:
    # ...
Tâche 4 : Mettre à jour la documentation
Modifie le fichier STATUS.md et/ou le JOURNAL-DEVELOPPEMENT.md pour marquer la Phase 0 comme "✅ COMPLÉTÉE" ou "TERMINÉE".

3. Critères d'Acceptation
Le script validate_piper.ps1 s'exécute avec succès.
Le script test_tts_handler.py corrigé s'exécute avec succès.
Le script run_assistant.py exécute la boucle complète ET affiche un rapport de latence détaillé après chaque tour.
La latence totale mesurée est inférieure à 1.2 secondes.