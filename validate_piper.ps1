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