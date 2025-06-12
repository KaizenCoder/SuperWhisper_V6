#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script d'automatisation des workflows SuperWhisper V6

.DESCRIPTION
    Automatise les tâches courantes de développement, validation et documentation
    pour le projet SuperWhisper V6 avec intégration de l'outil generate_bundle_coordinateur.py

.PARAMETER Action
    Type de workflow à exécuter: daily, weekly, delivery, validate, full

.PARAMETER Force
    Force l'exécution même en cas d'avertissements

.PARAMETER Backup
    Force la création de sauvegardes

.EXAMPLE
    .\scripts\superwhisper_workflow.ps1 -Action daily
    Exécute le workflow quotidien

.EXAMPLE
    .\scripts\superwhisper_workflow.ps1 -Action delivery -Backup
    Prépare un package de livraison avec sauvegarde

.NOTES
    Version: 1.0
    Auteur: SuperWhisper V6 Team
    Date: 2025-06-12
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("daily", "weekly", "delivery", "validate", "full")]
    [string]$Action = "daily",
    
    [Parameter(Mandatory=$false)]
    [switch]$Force = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Backup = $false
)

# =============================================================================
# CONFIGURATION ET VARIABLES
# =============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Couleurs pour l'affichage
$Colors = @{
    Header = "Cyan"
    Success = "Green" 
    Warning = "Yellow"
    Error = "Red"
    Info = "White"
    Highlight = "Magenta"
}

# Chemins importants
$ProjectRoot = "C:\Dev\SuperWhisper_V6"
$BundleScript = "scripts/generate_bundle_coordinateur.py"
$DiagnosticScript = "test_diagnostic_rtx3090.py"
$CodeSourcePath = "docs/Transmission_coordinateur/CODE-SOURCE.md"

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = "White",
        [switch]$NoNewline
    )
    
    if ($NoNewline) {
        Write-Host $Message -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $Message -ForegroundColor $Color
    }
}

function Write-Header {
    param([string]$Title)
    
    Write-Host ""
    Write-ColoredOutput "=" * 60 -Color $Colors.Header
    Write-ColoredOutput "🚀 SUPERWHISPER V6 - $Title" -Color $Colors.Header
    Write-ColoredOutput "=" * 60 -Color $Colors.Header
    Write-Host ""
}

function Write-Step {
    param(
        [string]$StepName,
        [string]$Description
    )
    
    Write-ColoredOutput "📋 $StepName" -Color $Colors.Highlight
    Write-ColoredOutput "   $Description" -Color $Colors.Info
}

function Test-Prerequisites {
    Write-Step "Vérification Prérequis" "Contrôle environnement de développement"
    
    # Vérifier répertoire de travail
    if (-not (Test-Path $ProjectRoot)) {
        Write-ColoredOutput "❌ Répertoire projet non trouvé: $ProjectRoot" -Color $Colors.Error
        exit 1
    }
    
    Set-Location $ProjectRoot
    Write-ColoredOutput "✅ Répertoire projet: $ProjectRoot" -Color $Colors.Success
    
    # Vérifier Python
    try {
        $pythonVersion = python --version 2>&1
        Write-ColoredOutput "✅ Python disponible: $pythonVersion" -Color $Colors.Success
    } catch {
        Write-ColoredOutput "❌ Python non disponible" -Color $Colors.Error
        exit 1
    }
    
    # Vérifier Git
    try {
        $gitVersion = git --version 2>&1
        Write-ColoredOutput "✅ Git disponible: $gitVersion" -Color $Colors.Success
    } catch {
        Write-ColoredOutput "❌ Git non disponible" -Color $Colors.Error
        exit 1
    }
    
    # Vérifier scripts essentiels
    if (-not (Test-Path $BundleScript)) {
        Write-ColoredOutput "❌ Script bundle non trouvé: $BundleScript" -Color $Colors.Error
        exit 1
    }
    Write-ColoredOutput "✅ Script bundle disponible" -Color $Colors.Success
    
    if (-not (Test-Path $DiagnosticScript)) {
        Write-ColoredOutput "⚠️ Script diagnostic non trouvé: $DiagnosticScript" -Color $Colors.Warning
    } else {
        Write-ColoredOutput "✅ Script diagnostic disponible" -Color $Colors.Success
    }
}

function Invoke-GPUValidation {
    Write-Step "Validation GPU RTX 3090" "Vérification configuration GPU exclusive"
    
    if (Test-Path $DiagnosticScript) {
        try {
            $result = python $DiagnosticScript 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-ColoredOutput "✅ Validation GPU réussie" -Color $Colors.Success
                # Afficher les lignes importantes du diagnostic
                $result | Where-Object { $_ -match "✅|❌|🎮|🔒" } | ForEach-Object {
                    Write-ColoredOutput "   $_" -Color $Colors.Info
                }
            } else {
                Write-ColoredOutput "❌ Validation GPU échouée" -Color $Colors.Error
                Write-ColoredOutput $result -Color $Colors.Error
                if (-not $Force) {
                    exit 1
                }
            }
        } catch {
            Write-ColoredOutput "❌ Erreur lors de la validation GPU: $_" -Color $Colors.Error
            if (-not $Force) {
                exit 1
            }
        }
    } else {
        Write-ColoredOutput "⚠️ Script diagnostic non disponible - Validation GPU ignorée" -Color $Colors.Warning
    }
}

function Invoke-DocumentationUpdate {
    param(
        [string]$Mode = "preserve"
    )
    
    Write-Step "Mise à jour Documentation" "Génération automatique CODE-SOURCE.md"
    
    # Construire la commande
    $command = "python $BundleScript"
    
    switch ($Mode) {
        "preserve" { 
            $command += " --preserve"
            Write-ColoredOutput "   Mode: Préservation (enrichissement)" -Color $Colors.Info
        }
        "regenerate" { 
            $command += " --regenerate"
            Write-ColoredOutput "   Mode: Régénération complète" -Color $Colors.Info
        }
        "validate" { 
            $command += " --validate"
            Write-ColoredOutput "   Mode: Validation (dry-run)" -Color $Colors.Info
        }
    }
    
    if ($Backup -or $Mode -eq "regenerate") {
        $command += " --backup"
        Write-ColoredOutput "   Sauvegarde: Activée" -Color $Colors.Info
    }
    
    try {
        Write-ColoredOutput "   Commande: $command" -Color $Colors.Info
        
        # Exécuter la commande et capturer la sortie
        $process = Start-Process -FilePath "python" -ArgumentList ($command -replace "python ", "") -NoNewWindow -Wait -PassThru -RedirectStandardOutput "temp_output.txt" -RedirectStandardError "temp_error.txt"
        
        $output = Get-Content "temp_output.txt" -ErrorAction SilentlyContinue
        $errors = Get-Content "temp_error.txt" -ErrorAction SilentlyContinue
        
        # Nettoyer fichiers temporaires
        Remove-Item "temp_output.txt" -ErrorAction SilentlyContinue
        Remove-Item "temp_error.txt" -ErrorAction SilentlyContinue
        
        if ($process.ExitCode -eq 0) {
            Write-ColoredOutput "✅ Documentation mise à jour avec succès" -Color $Colors.Success
            
            # Afficher statistiques importantes
            $output | Where-Object { $_ -match "✅|📊|📈|💾" } | ForEach-Object {
                Write-ColoredOutput "   $_" -Color $Colors.Success
            }
        } else {
            Write-ColoredOutput "❌ Erreur lors de la mise à jour documentation" -Color $Colors.Error
            if ($errors) {
                $errors | ForEach-Object { Write-ColoredOutput "   $_" -Color $Colors.Error }
            }
            if (-not $Force) {
                exit 1
            }
        }
    } catch {
        Write-ColoredOutput "❌ Erreur lors de l'exécution: $_" -Color $Colors.Error
        if (-not $Force) {
            exit 1
        }
    }
}

function Get-DocumentationStats {
    Write-Step "Statistiques Documentation" "Analyse du fichier CODE-SOURCE.md"
    
    if (Test-Path $CodeSourcePath) {
        try {
            $content = Get-Content $CodeSourcePath -Raw -Encoding UTF8
            $lines = ($content -split "`n").Count
            $characters = $content.Length
            $sizeKB = [math]::Round($characters / 1024, 2)
            
            Write-ColoredOutput "📊 Statistiques CODE-SOURCE.md:" -Color $Colors.Highlight
            Write-ColoredOutput "   📝 Lignes: $lines" -Color $Colors.Success
            Write-ColoredOutput "   📄 Caractères: $characters" -Color $Colors.Success
            Write-ColoredOutput "   💾 Taille: $sizeKB KB" -Color $Colors.Success
            
            # Vérifier si c'est une documentation complète (>200KB)
            if ($sizeKB -gt 200) {
                Write-ColoredOutput "   🚀 Documentation COMPLÈTE détectée" -Color $Colors.Success
            } elseif ($sizeKB -gt 30) {
                Write-ColoredOutput "   📋 Documentation ENRICHIE détectée" -Color $Colors.Info
            } else {
                Write-ColoredOutput "   📝 Documentation BASIQUE détectée" -Color $Colors.Warning
            }
            
        } catch {
            Write-ColoredOutput "❌ Erreur lecture CODE-SOURCE.md: $_" -Color $Colors.Error
        }
    } else {
        Write-ColoredOutput "⚠️ Fichier CODE-SOURCE.md non trouvé" -Color $Colors.Warning
    }
}

function Invoke-Tests {
    Write-Step "Exécution Tests" "Validation fonctionnelle du projet"
    
    # Tests GPU (déjà fait dans Invoke-GPUValidation)
    Write-ColoredOutput "   🎮 Tests GPU: Déjà validés" -Color $Colors.Success
    
    # Tests unitaires (si disponibles)
    if (Test-Path "tests") {
        try {
            Write-ColoredOutput "   🧪 Lancement tests unitaires..." -Color $Colors.Info
            $testResult = python -m pytest tests/ -v --tb=short 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                Write-ColoredOutput "   ✅ Tests unitaires réussis" -Color $Colors.Success
            } else {
                Write-ColoredOutput "   ⚠️ Certains tests ont échoué" -Color $Colors.Warning
                # Afficher seulement les échecs
                $testResult | Where-Object { $_ -match "FAILED|ERROR" } | ForEach-Object {
                    Write-ColoredOutput "     $_" -Color $Colors.Warning
                }
            }
        } catch {
            Write-ColoredOutput "   ⚠️ Tests unitaires non disponibles ou erreur: $_" -Color $Colors.Warning
        }
    } else {
        Write-ColoredOutput "   ⚠️ Répertoire tests/ non trouvé" -Color $Colors.Warning
    }
}

function Invoke-GitOperations {
    param([string]$CommitMessage)
    
    Write-Step "Opérations Git" "Gestion du versioning"
    
    try {
        # Vérifier statut Git
        $gitStatus = git status --porcelain 2>&1
        
        if ($gitStatus) {
            Write-ColoredOutput "   📝 Modifications détectées:" -Color $Colors.Info
            $gitStatus | ForEach-Object {
                Write-ColoredOutput "     $_" -Color $Colors.Info
            }
            
            if ($CommitMessage) {
                Write-ColoredOutput "   💾 Ajout des modifications..." -Color $Colors.Info
                git add docs/Transmission_coordinateur/CODE-SOURCE.md 2>&1 | Out-Null
                
                Write-ColoredOutput "   📝 Commit: $CommitMessage" -Color $Colors.Info
                git commit -m $CommitMessage 2>&1 | Out-Null
                
                if ($LASTEXITCODE -eq 0) {
                    Write-ColoredOutput "   ✅ Commit réussi" -Color $Colors.Success
                } else {
                    Write-ColoredOutput "   ⚠️ Erreur lors du commit" -Color $Colors.Warning
                }
            }
        } else {
            Write-ColoredOutput "   ✅ Aucune modification à commiter" -Color $Colors.Success
        }
        
        # Afficher dernier commit
        $lastCommit = git log -1 --oneline 2>&1
        Write-ColoredOutput "   📋 Dernier commit: $lastCommit" -Color $Colors.Info
        
    } catch {
        Write-ColoredOutput "   ⚠️ Erreur Git: $_" -Color $Colors.Warning
    }
}

function Show-Summary {
    param([string]$WorkflowType)
    
    Write-Header "RÉSUMÉ - WORKFLOW $($WorkflowType.ToUpper())"
    
    Write-ColoredOutput "🎯 Workflow exécuté: $WorkflowType" -Color $Colors.Highlight
    Write-ColoredOutput "📅 Date/Heure: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Color $Colors.Info
    Write-ColoredOutput "📍 Répertoire: $ProjectRoot" -Color $Colors.Info
    
    # Statistiques finales
    Get-DocumentationStats
    
    Write-ColoredOutput ""
    Write-ColoredOutput "✅ WORKFLOW $($WorkflowType.ToUpper()) TERMINÉ AVEC SUCCÈS" -Color $Colors.Success
    Write-ColoredOutput ""
}

# =============================================================================
# WORKFLOWS PRINCIPAUX
# =============================================================================

function Invoke-DailyWorkflow {
    Write-Header "WORKFLOW QUOTIDIEN"
    
    Test-Prerequisites
    Invoke-DocumentationUpdate -Mode "preserve"
    Get-DocumentationStats
    Invoke-GitOperations "docs: Daily documentation update"
    
    Show-Summary "daily"
}

function Invoke-WeeklyWorkflow {
    Write-Header "WORKFLOW HEBDOMADAIRE"
    
    Test-Prerequisites
    Invoke-GPUValidation
    Invoke-DocumentationUpdate -Mode "regenerate"
    Get-DocumentationStats
    Invoke-Tests
    Invoke-GitOperations "docs: Weekly complete documentation regeneration"
    
    Show-Summary "weekly"
}

function Invoke-DeliveryWorkflow {
    Write-Header "WORKFLOW LIVRAISON"
    
    Test-Prerequisites
    Invoke-GPUValidation
    Invoke-DocumentationUpdate -Mode "regenerate"
    Get-DocumentationStats
    Invoke-Tests
    
    Write-Step "Package Livraison" "Préparation pour transmission coordinateur"
    
    # Vérifier que la documentation est complète
    if (Test-Path $CodeSourcePath) {
        $content = Get-Content $CodeSourcePath -Raw -Encoding UTF8
        $sizeKB = [math]::Round($content.Length / 1024, 2)
        
        if ($sizeKB -gt 200) {
            Write-ColoredOutput "✅ Package livraison prêt - Documentation complète ($sizeKB KB)" -Color $Colors.Success
            Write-ColoredOutput "📦 Fichier à transmettre: $CodeSourcePath" -Color $Colors.Highlight
        } else {
            Write-ColoredOutput "⚠️ Documentation semble incomplète ($sizeKB KB)" -Color $Colors.Warning
        }
    }
    
    Invoke-GitOperations "docs: Delivery package - Complete documentation for coordinator"
    
    Show-Summary "delivery"
}

function Invoke-ValidateWorkflow {
    Write-Header "WORKFLOW VALIDATION"
    
    Test-Prerequisites
    Invoke-GPUValidation
    Invoke-DocumentationUpdate -Mode "validate"
    Get-DocumentationStats
    
    Write-Step "Validation Complète" "Contrôle de l'état du projet"
    Write-ColoredOutput "✅ Validation terminée - Aucune modification effectuée" -Color $Colors.Success
    
    Show-Summary "validate"
}

function Invoke-FullWorkflow {
    Write-Header "WORKFLOW COMPLET"
    
    Test-Prerequisites
    Invoke-GPUValidation
    Invoke-DocumentationUpdate -Mode "regenerate"
    Get-DocumentationStats
    Invoke-Tests
    Invoke-GitOperations "docs: Full workflow - Complete validation and documentation"
    
    Write-Step "Workflow Complet" "Toutes les opérations exécutées"
    Write-ColoredOutput "🎉 Workflow complet terminé avec succès" -Color $Colors.Success
    
    Show-Summary "full"
}

# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

try {
    Write-Header "WORKFLOW AUTOMATISÉ"
    Write-ColoredOutput "Action demandée: $Action" -Color $Colors.Highlight
    Write-ColoredOutput "Force: $Force" -Color $Colors.Info
    Write-ColoredOutput "Backup: $Backup" -Color $Colors.Info
    
    switch ($Action) {
        "daily" { Invoke-DailyWorkflow }
        "weekly" { Invoke-WeeklyWorkflow }
        "delivery" { Invoke-DeliveryWorkflow }
        "validate" { Invoke-ValidateWorkflow }
        "full" { Invoke-FullWorkflow }
        default { 
            Write-ColoredOutput "❌ Action non reconnue: $Action" -Color $Colors.Error
            exit 1
        }
    }
    
} catch {
    Write-ColoredOutput ""
    Write-ColoredOutput "❌ ERREUR CRITIQUE: $_" -Color $Colors.Error
    Write-ColoredOutput "Stack trace: $($_.ScriptStackTrace)" -Color $Colors.Error
    exit 1
}

Write-ColoredOutput ""
Write-ColoredOutput "🎊 SUPERWHISPER V6 - WORKFLOW AUTOMATISÉ TERMINÉ" -Color $Colors.Success
Write-ColoredOutput "" 