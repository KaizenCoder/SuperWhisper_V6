# Test-PiperVoice.ps1 (script propre)
# Script de téléchargement et test du modèle Piper français fr_FR-upmc-medium

# --- CONFIGURATION ---
# Chemin vers le dossier des modèles
$ModelDir = "models"
# Nom du modèle (utilisé pour les noms de fichiers)
$ModelName = "fr_FR-upmc-medium"

# URL de base du modèle sur Hugging Face
$BaseURL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium"

# Noms complets des fichiers requis
$OnnxFileName = "$ModelName.onnx"
$JsonFileName = "$ModelName.onnx.json"

# Chemins locaux complets
$LocalOnnxPath = Join-Path $ModelDir $OnnxFileName
$LocalJsonPath = Join-Path $ModelDir $JsonFileName

Write-Host "🇫🇷 TÉLÉCHARGEMENT MODÈLE PIPER FRANÇAIS CORRIGÉ" -ForegroundColor Green
Write-Host "=" * 60
Write-Host "📦 Modèle: $ModelName"
Write-Host "🔗 Base URL: $BaseURL"
Write-Host "📁 Dossier local: $ModelDir"
Write-Host ""

# --- LOGIQUE ---

# Crée le dossier 'models' si besoin
if (-not (Test-Path $ModelDir)) {
    Write-Host "📂 Création du dossier '$ModelDir'..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $ModelDir | Out-Null
    Write-Host "✅ Dossier créé" -ForegroundColor Green
}

# 1. Téléchargement du fichier .onnx si absent
if (-not (Test-Path $LocalOnnxPath)) {
    $OnnxUrl = "$BaseURL/$OnnxFileName"
    Write-Host "⬇️ Téléchargement du modèle .onnx depuis $OnnxUrl" -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri $OnnxUrl -OutFile $LocalOnnxPath -UseBasicParsing
        Write-Host "✅ Modèle .onnx téléchargé." -ForegroundColor Green
        
        # Vérifier la taille du fichier
        $onnxSize = (Get-Item $LocalOnnxPath).Length
        Write-Host "📊 Taille fichier ONNX: $([math]::Round($onnxSize/1MB, 2)) MB"
    } catch {
        Write-Error "❌ ERREUR lors du téléchargement du fichier .onnx. Vérifiez l'URL ou votre connexion."
        Write-Host "🔗 URL tentée: $OnnxUrl"
        exit 1
    }
} else {
    Write-Host "👍 Le fichier .onnx existe déjà." -ForegroundColor Green
}

# 2. Téléchargement du fichier .onnx.json si absent
if (-not (Test-Path $LocalJsonPath)) {
    $JsonUrl = "$BaseURL/$JsonFileName"
    Write-Host "⬇️ Téléchargement de la configuration .json depuis $JsonUrl" -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri $JsonUrl -OutFile $LocalJsonPath -UseBasicParsing
        Write-Host "✅ Configuration .json téléchargée." -ForegroundColor Green
        
        # Vérifier le contenu JSON
        $jsonContent = Get-Content $LocalJsonPath | ConvertFrom-Json
        Write-Host "📊 Langue détectée: $($jsonContent.language)"
        Write-Host "📊 Sample rate: $($jsonContent.sample_rate)Hz"
    } catch {
        Write-Error "❌ ERREUR lors du téléchargement du fichier .json."
        Write-Host "🔗 URL tentée: $JsonUrl"
        exit 1
    }
} else {
    Write-Host "👍 Le fichier .json existe déjà." -ForegroundColor Green
}

# 3. Vérification de la présence des deux fichiers
if ((Test-Path $LocalOnnxPath) -and (Test-Path $LocalJsonPath)) {
    Write-Host ""
    Write-Host "✅ Modèle et configuration détectés : $ModelName" -ForegroundColor Green
    Write-Host "   📄 ONNX: $LocalOnnxPath"
    Write-Host "   📄 JSON: $LocalJsonPath"
} else {
    Write-Error "❌ Fichiers requis manquants pour $ModelName. Impossible de continuer."
    exit 1
}

# 4. Test de la synthèse vocale
Write-Host ""
Write-Host "🎤 TEST SYNTHÈSE VOCALE FRANÇAISE" -ForegroundColor Green
Write-Host "=" * 40

$piperCmd = "piper" # Assurez-vous que 'piper' est dans votre PATH
$text = "Bonjour, ceci est un test vocal avec le modèle UPMC correctement téléchargé depuis Hugging Face."

Write-Host "📝 Texte: '$text'"
Write-Host "🔧 Commande Piper: $piperCmd"
Write-Host "🔊 Génération du fichier de test 'test.wav'..." -ForegroundColor Yellow

# Appelle Piper pour générer un fichier wav
# La commande 'echo' peut avoir des soucis d'encodage, Set-Content est plus sûr
try {
    Set-Content -Path "input.txt" -Value $text -Encoding UTF8
    Get-Content "input.txt" | & $piperCmd `
        --model $LocalOnnxPath `
        --output_file "test.wav"

    if (Test-Path "test.wav") {
        Write-Host "✅ Fichier test.wav généré avec succès." -ForegroundColor Green
        
        # Informations sur le fichier généré
        $wavFile = Get-Item "test.wav"
        Write-Host "📊 Taille: $($wavFile.Length) bytes"
        Write-Host "📅 Créé: $($wavFile.CreationTime)"
        
        Write-Host ""
        Write-Host "🎧 LECTURE AUDIO:" -ForegroundColor Yellow
        Write-Host "   Pour écouter: Invoke-Item test.wav"
        
        # Optionnel: jouer le son automatiquement
        # (New-Object System.Media.SoundPlayer("test.wav")).PlaySync()
        
    } else {
        Write-Error "❌ La génération du fichier test.wav a échoué."
        exit 1
    }
} catch {
    Write-Error "❌ Erreur lors de la synthèse Piper: $_"
    Write-Host "💡 Vérifiez que 'piper' est installé et accessible dans PATH"
    Write-Host "💡 Ou que l'environnement venv_piper312 est activé"
    exit 1
}

# Nettoyage
if (Test-Path "input.txt") {
    Remove-Item "input.txt"
}

Write-Host ""
Write-Host "🎉 TEST PIPER FRANÇAIS COMPLÉTÉ !" -ForegroundColor Green
Write-Host "✅ Modèle fr_FR-upmc-medium opérationnel"
Write-Host "📁 Fichiers dans: $ModelDir"
Write-Host "🎵 Audio test: test.wav"
Write-Host "🔄 Ce modèle peut maintenant être utilisé pour remplacer SAPI" 