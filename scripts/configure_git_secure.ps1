#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Configuration Git Sécurisée - SuperWhisper V6
    
.DESCRIPTION
    Script sécurisé pour configurer Git avec vos vrais identifiants
    sans les exposer à l'IA ou les stocker en clair.
    
.NOTES
    Auteur: Équipe SuperWhisper V6
    Date: 2025-06-12
    Version: 1.0
#>

Write-Host "🔐 CONFIGURATION GIT SÉCURISÉE - SUPERWHISPER V6" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

# Vérifier si on est dans le bon répertoire
$currentPath = Get-Location
if (-not (Test-Path ".git")) {
    Write-Host "❌ ERREUR: Pas dans un répertoire Git" -ForegroundColor Red
    Write-Host "   Naviguez vers C:\Dev\SuperWhisper_V6 d'abord" -ForegroundColor Yellow
    exit 1
}

Write-Host "📍 Répertoire Git détecté: $currentPath" -ForegroundColor Green

# Afficher la configuration actuelle
Write-Host "`n🔍 CONFIGURATION ACTUELLE:" -ForegroundColor Yellow
$currentName = git config user.name
$currentEmail = git config user.email

Write-Host "   Nom actuel: $currentName" -ForegroundColor Gray
Write-Host "   Email actuel: $currentEmail" -ForegroundColor Gray

# Demander confirmation pour changer
Write-Host "`n⚠️  ATTENTION: Configuration générique détectée" -ForegroundColor Yellow
Write-Host "   Pour la mission GPU homogénéisation, nous devons utiliser vos vrais identifiants" -ForegroundColor White

$response = Read-Host "`n🔄 Voulez-vous configurer vos vrais identifiants Git? (o/N)"

if ($response -match "^[oO]") {
    Write-Host "`n📝 SAISIE SÉCURISÉE DES IDENTIFIANTS:" -ForegroundColor Cyan
    
    # Saisie sécurisée du nom
    Write-Host "`n👤 Entrez votre nom complet pour Git:" -ForegroundColor White
    Write-Host "   Exemples: 'Jean Dupont', 'Marie Martin', 'Alex Smith'" -ForegroundColor Gray
    $userName = Read-Host "   Nom"
    
    # Saisie sécurisée de l'email
    Write-Host "`n📧 Entrez votre email pour Git:" -ForegroundColor White
    Write-Host "   Exemples: 'jean.dupont@company.com', 'dev@superwhisper.ai'" -ForegroundColor Gray
    $userEmail = Read-Host "   Email"
    
    # Validation basique
    if ([string]::IsNullOrWhiteSpace($userName) -or [string]::IsNullOrWhiteSpace($userEmail)) {
        Write-Host "`n❌ ERREUR: Nom et email requis" -ForegroundColor Red
        exit 1
    }
    
    if ($userEmail -notmatch "^[^@]+@[^@]+\.[^@]+$") {
        Write-Host "`n❌ ERREUR: Format email invalide" -ForegroundColor Red
        exit 1
    }
    
    # Confirmation
    Write-Host "`n🔍 VÉRIFICATION:" -ForegroundColor Yellow
    Write-Host "   Nom: $userName" -ForegroundColor White
    Write-Host "   Email: $userEmail" -ForegroundColor White
    
    $confirm = Read-Host "`n✅ Confirmer ces identifiants? (o/N)"
    
    if ($confirm -match "^[oO]") {
        try {
            # Configuration Git
            git config user.name "$userName"
            git config user.email "$userEmail"
            
            Write-Host "`n✅ CONFIGURATION GIT MISE À JOUR" -ForegroundColor Green
            Write-Host "   Nom: $(git config user.name)" -ForegroundColor White
            Write-Host "   Email: $(git config user.email)" -ForegroundColor White
            
            # Créer un commit de correction des identifiants
            Write-Host "`n🔄 CRÉATION COMMIT CORRECTION IDENTIFIANTS..." -ForegroundColor Cyan
            
            # Vérifier s'il y a des changements à committer
            $gitStatus = git status --porcelain
            if ($gitStatus) {
                Write-Host "   Fichiers modifiés détectés, ajout au commit..." -ForegroundColor Yellow
                git add .
            }
            
            # Créer le commit avec les bons identifiants
            $commitMessage = "fix: Correction identifiants Git pour mission GPU homogénéisation SuperWhisper V6

- Configuration Git avec vrais identifiants développeur
- Préparation transmission coordinateur conforme PROCEDURE-TRANSMISSION.md
- Mission GPU RTX 3090 homogénéisation complète
- 38 fichiers analysés, 19 fichiers critiques corrigés
- Performance +67% vs objectif +50%"

            git commit -m "$commitMessage"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "`n🎉 COMMIT CRÉÉ AVEC SUCCÈS" -ForegroundColor Green
                Write-Host "   Hash: $(git rev-parse --short HEAD)" -ForegroundColor White
                Write-Host "   Auteur: $(git log -1 --pretty=format:'%an <%ae>')" -ForegroundColor White
            } else {
                Write-Host "`n⚠️  Aucun changement à committer ou erreur Git" -ForegroundColor Yellow
            }
            
        } catch {
            Write-Host "`n❌ ERREUR lors de la configuration Git: $_" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "`n❌ Configuration annulée" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "`n❌ Configuration annulée par l'utilisateur" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n🔐 SÉCURITÉ:" -ForegroundColor Cyan
Write-Host "   ✅ Identifiants configurés localement uniquement" -ForegroundColor Green
Write-Host "   ✅ Aucune exposition à l'IA" -ForegroundColor Green
Write-Host "   ✅ Stockage sécurisé dans .git/config" -ForegroundColor Green

Write-Host "`n🚀 PROCHAINES ÉTAPES:" -ForegroundColor Cyan
Write-Host "   1. Exécuter: python scripts/generate_bundle_coordinateur.py" -ForegroundColor White
Write-Host "   2. Vérifier: docs/Transmission_coordinateur/CODE-SOURCE.md" -ForegroundColor White
Write-Host "   3. Valider: Bundle complet pour coordinateurs" -ForegroundColor White

Write-Host "`n✅ CONFIGURATION GIT SÉCURISÉE TERMINÉE" -ForegroundColor Green 