# Script de sauvegarde automatique des 40 fichiers à corriger
# Mission : Homogénéisation GPU SuperWhisper V6

Write-Host "🚀 DÉMARRAGE SAUVEGARDE - 40 fichiers pour homogénéisation GPU" -ForegroundColor Green

# Liste des 40 fichiers à corriger
$filesToBackup = @(
    # Modules Core Critiques (7)
    "benchmarks/benchmark_stt_realistic.py",
    "LLM/llm_manager_enhanced.py",
    "LUXA_TTS/tts_handler_coqui.py",
    "Orchestrator/fallback_manager.py",
    "STT/vad_manager_optimized.py",
    "TTS/tts_handler_coqui.py",
    "TTS/tts_handler_piper_native.py",
    
    # Modules Core Supplémentaires (6)
    "STT/stt_manager_robust.py",
    "STT/vad_manager.py",
    "TTS/tts_handler_piper_espeak.py",
    "TTS/tts_handler_piper_fixed.py",
    "TTS/tts_handler_piper_french.py",
    "utils/gpu_manager.py",
    
    # Scripts Test Initiaux (13)
    "tests/test_double_check_corrections.py",
    "tests/test_double_check_validation_simple.py",
    "test_cuda_debug.py",
    "test_cuda.py",
    "test_espeak_french.py",
    "test_french_voice.py",
    "test_gpu_correct.py",
    "test_piper_native.py",
    "test_tts_fixed.py",
    "test_tts_long_feedback.py",
    "test_upmc_model.py",
    "test_validation_decouverte.py",
    "TTS/tts_handler_piper_rtx3090.py",
    
    # Scripts Supplémentaires (2)
    "tests/test_llm_handler.py",
    "tests/test_stt_handler.py",
    
    # Scripts Validation Exhaustifs (12)
    "test_correction_validation_1.py",
    "test_correction_validation_2.py",
    "test_correction_validation_3.py",
    "test_correction_validation_4.py",
    "test_rtx3090_detection.py",
    "test_tts_rtx3090_performance.py",
    "test_validation_globale_finale.py",
    "test_validation_mvp_settings.py",
    "test_validation_rtx3090_detection.py",
    "test_validation_stt_manager_robust.py",
    "test_validation_tts_performance.py",
    "validate_gpu_config.py"
)

$backupDir = "docs/gpu-correction/backups"
$successCount = 0
$errorCount = 0
$notFoundCount = 0

Write-Host "📁 Dossier de sauvegarde : $backupDir" -ForegroundColor Cyan
Write-Host "📊 Nombre total de fichiers : $($filesToBackup.Count)" -ForegroundColor Cyan
Write-Host ""

foreach ($file in $filesToBackup) {
    $sourceFile = $file.Replace('/', '\')
    $fileName = Split-Path $sourceFile -Leaf
    $backupFile = Join-Path $backupDir "$fileName.backup"
    
    Write-Host "🔄 Sauvegarde : $sourceFile" -NoNewline
    
    if (Test-Path $sourceFile) {
        try {
            Copy-Item $sourceFile $backupFile -Force
            Write-Host " ✅ OK" -ForegroundColor Green
            $successCount++
        }
        catch {
            Write-Host " ❌ ERREUR : $_" -ForegroundColor Red
            $errorCount++
        }
    }
    else {
        Write-Host " ⚠️ INTROUVABLE" -ForegroundColor Yellow
        $notFoundCount++
    }
}

Write-Host ""
Write-Host "📈 RÉSUMÉ SAUVEGARDE :" -ForegroundColor Green
Write-Host "  ✅ Réussis : $successCount" -ForegroundColor Green
Write-Host "  ❌ Erreurs : $errorCount" -ForegroundColor Red
Write-Host "  ⚠️ Introuvables : $notFoundCount" -ForegroundColor Yellow
Write-Host "  📊 Total : $($filesToBackup.Count)" -ForegroundColor Cyan

if ($errorCount -eq 0 -and $notFoundCount -eq 0) {
    Write-Host ""
    Write-Host "🎯 SAUVEGARDE COMPLÈTE RÉUSSIE !" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "🚨 SAUVEGARDE INCOMPLÈTE - Vérifier les erreurs" -ForegroundColor Red
    exit 1
} 