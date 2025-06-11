<#
Skript : build_piper_312.ps1
But      : Compiler et installer piper-tts pour Python 3.12 avec support GPU (onnxruntime-gpu)
#>

$ErrorActionPreference = "Stop"

Write-Host "🔧 Installation prérequis (Rust, BuildTools, CMake)…"

# 1) Rust toolchain
if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
    winget install --id Rustlang.Rustup -e --source winget
}

# 2) Visual Studio Build Tools 2022 (C++ toolchain)
$vsPath = "C:\BuildTools"
if (-not (Test-Path $vsPath)) {
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
      --override "--installPath $vsPath --add Microsoft.VisualStudio.Workload.VCTools --quiet --wait --norestart"
}

# 3) CMake
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    winget install --id Kitware.CMake -e --source winget
}

# 4) Créer venv Python 3.12
Write-Host "🐍 Création venv Python 3.12…"
python -m venv venv_piper312
& "venv_piper312\Scripts\Activate.ps1"

python -m pip install -U pip wheel setuptools maturin
python -m pip install onnxruntime-gpu numpy  # dépendances runtime

# 5) Cloner et compiler piper-tts
Write-Host "📥 Clone piper-tts…"
git clone https://github.com/rhasspy/piper.git
Set-Location piper

Write-Host "🔨 Compilation wheel via maturin… (quelques minutes)"
maturin pep517 build --release -i ..\venv_piper312\Scripts\python.exe

# 6) Installation de la wheel générée
$wheel = Get-ChildItem -Path "target\wheels" -Filter "piper_tts-*.whl" | Select-Object -First 1
python -m pip install $wheel.FullName

# 7) Installation d'une voix FR et test GPU
Write-Host "🎙️ Téléchargement voix fr_FR-siwis-medium…"
piper-tts-download --model fr_FR-siwis-medium --output .\models

Write-Host "▶️ Test de synthèse GPU…"
"Bonjour, je suis LUXA, votre assistant vocal local." |
    piper --model .\models\fr_FR-siwis-medium.onnx --output_file test.wav --use_gpu

Write-Host "✅ Terminé. Fichier 'test.wav' généré." 