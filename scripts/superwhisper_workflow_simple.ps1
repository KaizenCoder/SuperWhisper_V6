#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script d'automatisation simplifié SuperWhisper V6

.DESCRIPTION
    Version simplifiée du workflow automatisé pour éviter les problèmes d'encodage

.PARAMETER Action
    Type de workflow: daily, weekly, delivery, validate

.EXAMPLE
    .\scripts\superwhisper_workflow_simple.ps1 -Action daily
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("daily", "weekly", "delivery", "validate")]
    [string]$Action = "daily"
)

# Configuration
$ProjectRoot = "C:\Dev\SuperWhisper_V6"
$BundleScript = "scripts/generate_bundle_coordinateur.py"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "SUPERWHISPER V6 - WORKFLOW AUTOMATISE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Action: $Action" -ForegroundColor Yellow
Write-Host ""

# Vérifier répertoire
if (-not (Test-Path $ProjectRoot)) {
    Write-Host "ERREUR: Répertoire projet non trouvé: $ProjectRoot" -ForegroundColor Red
    exit 1
}

Set-Location $ProjectRoot
Write-Host "Répertoire projet: $ProjectRoot" -ForegroundColor Green

# Vérifier Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python disponible: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERREUR: Python non disponible" -ForegroundColor Red
    exit 1
}

# Vérifier script bundle
if (-not (Test-Path $BundleScript)) {
    Write-Host "ERREUR: Script bundle non trouvé: $BundleScript" -ForegroundColor Red
    exit 1
}
Write-Host "Script bundle disponible" -ForegroundColor Green

Write-Host ""
Write-Host "Exécution workflow: $Action" -ForegroundColor Cyan

switch ($Action) {
    "daily" {
        Write-Host "Workflow quotidien - Mise à jour documentation..." -ForegroundColor Yellow
        python $BundleScript --preserve
    }
    "weekly" {
        Write-Host "Workflow hebdomadaire - Régénération complète..." -ForegroundColor Yellow
        python $BundleScript --regenerate --backup
    }
    "delivery" {
        Write-Host "Workflow livraison - Package coordinateur..." -ForegroundColor Yellow
        python $BundleScript --regenerate --backup
        
        # Vérifier taille documentation
        if (Test-Path "docs/Transmission_coordinateur/CODE-SOURCE.md") {
            $content = Get-Content "docs/Transmission_coordinateur/CODE-SOURCE.md" -Raw -Encoding UTF8
            $sizeKB = [math]::Round($content.Length / 1024, 2)
            
            if ($sizeKB -gt 200) {
                Write-Host "Package livraison prêt - Documentation complète ($sizeKB KB)" -ForegroundColor Green
                Write-Host "Fichier à transmettre: docs/Transmission_coordinateur/CODE-SOURCE.md" -ForegroundColor Magenta
            } else {
                Write-Host "ATTENTION: Documentation semble incomplète ($sizeKB KB)" -ForegroundColor Yellow
            }
        }
    }
    "validate" {
        Write-Host "Workflow validation - Contrôle sans modification..." -ForegroundColor Yellow
        python $BundleScript --validate
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "WORKFLOW $($Action.ToUpper()) TERMINÉ AVEC SUCCÈS" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "ERREUR LORS DU WORKFLOW $($Action.ToUpper())" -ForegroundColor Red
    exit 1
}

# Statistiques finales
if (Test-Path "docs/Transmission_coordinateur/CODE-SOURCE.md") {
    Write-Host ""
    Write-Host "Statistiques Documentation:" -ForegroundColor Cyan
    
    $content = Get-Content "docs/Transmission_coordinateur/CODE-SOURCE.md" -Raw -Encoding UTF8
    $lines = ($content -split "`n").Count
    $characters = $content.Length
    $sizeKB = [math]::Round($characters / 1024, 2)
    
    Write-Host "  Lignes: $lines" -ForegroundColor White
    Write-Host "  Caractères: $characters" -ForegroundColor White
    Write-Host "  Taille: $sizeKB KB" -ForegroundColor White
    
    if ($sizeKB -gt 200) {
        Write-Host "  Type: Documentation COMPLÈTE" -ForegroundColor Green
    } elseif ($sizeKB -gt 30) {
        Write-Host "  Type: Documentation ENRICHIE" -ForegroundColor Yellow
    } else {
        Write-Host "  Type: Documentation BASIQUE" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "SUPERWHISPER V6 - WORKFLOW AUTOMATISE TERMINÉ" -ForegroundColor Green
Write-Host "" 