#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Configuration Git Simplifiée - SuperWhisper V6
    
.DESCRIPTION
    Version simplifiée pour configurer Git avec vos identifiants
    sans problème d'interaction dans le terminal.
    
.NOTES
    Auteur: Équipe SuperWhisper V6
    Date: 2025-06-12
    Version: 1.1 - Simplifiée
#>

Write-Host "🔐 CONFIGURATION GIT SIMPLIFIÉE - SUPERWHISPER V6" -ForegroundColor Cyan
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
$currentName = git config user.name
$currentEmail = git config user.email

Write-Host "`n🔍 CONFIGURATION ACTUELLE:" -ForegroundColor Yellow
Write-Host "   Nom actuel: $currentName" -ForegroundColor White
Write-Host "   Email actuel: $currentEmail" -ForegroundColor White

if ($currentName -eq "ModelesSuivi" -or $currentEmail -eq "modeles@example.com") {
    Write-Host "`n⚠️  ATTENTION: Configuration générique détectée" -ForegroundColor Yellow
    Write-Host "   Pour la mission GPU homogénéisation, nous devons utiliser vos vrais identifiants" -ForegroundColor Yellow
    
    Write-Host "`n📝 INSTRUCTIONS POUR CONFIGURATION MANUELLE:" -ForegroundColor Cyan
    Write-Host "   Exécutez ces commandes avec VOS vrais identifiants:" -ForegroundColor White
    Write-Host ""
    Write-Host "   git config user.name `"VOTRE_VRAI_NOM`"" -ForegroundColor Green
    Write-Host "   git config user.email `"votre.email@domaine.com`"" -ForegroundColor Green
    Write-Host ""
    Write-Host "   Exemple:" -ForegroundColor Gray
    Write-Host "   git config user.name `"Jean Dupont`"" -ForegroundColor Gray
    Write-Host "   git config user.email `"jean.dupont@entreprise.com`"" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🔒 SÉCURITÉ: Remplacez par VOS vrais identifiants" -ForegroundColor Red
    Write-Host "   Ne partagez jamais vos identifiants avec l'IA" -ForegroundColor Red
    
} else {
    Write-Host "`n✅ Configuration personnalisée détectée" -ForegroundColor Green
    Write-Host "   Nom: $currentName" -ForegroundColor White
    Write-Host "   Email: $currentEmail" -ForegroundColor White
}

Write-Host "`n🚀 PROCHAINES ÉTAPES:" -ForegroundColor Cyan
Write-Host "   1. Configurez Git avec vos vrais identifiants (commandes ci-dessus)" -ForegroundColor White
Write-Host "   2. Vérifiez: git config user.name && git config user.email" -ForegroundColor White
Write-Host "   3. Continuez avec: python scripts/generate_bundle_coordinateur.py --validate" -ForegroundColor White

Write-Host "`n✅ SCRIPT TERMINÉ - Configuration manuelle requise" -ForegroundColor Green 