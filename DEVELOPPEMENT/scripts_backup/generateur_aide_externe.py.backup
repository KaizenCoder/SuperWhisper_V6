#!/usr/bin/env python3
"""
Générateur d'aide externe SuperWhisper V6
Outil pour créer des demandes d'aide avec code essentiel agrégé en .md
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json

class GenerateurAideExterne:
    """Générateur automatisé de demandes d'aide externe"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def creer_aide_externe(self, 
                          probleme: str,
                          fichiers_critiques: List[str],
                          contexte: str = "",
                          urgence: str = "NORMALE",
                          titre_court: str = "AIDE_EXTERNE") -> Dict[str, str]:
        """Crée une demande d'aide externe complète"""
        
        print(f"🆘 Génération aide externe : {probleme}")
        print(f"📂 Fichiers critiques : {len(fichiers_critiques)}")
        print(f"⚡ Urgence : {urgence}")
        
        # 1. Collecte code essentiel
        code_essentiel = self._collecter_code_essentiel(fichiers_critiques)
        
        # 2. Génération document .md principal
        doc_principal = self._generer_document_principal(
            probleme, code_essentiel, contexte, urgence
        )
        
        # 3. Création fichiers
        nom_fichier = f"{titre_court}_{self.timestamp}.md"
        chemin_principal = self.project_root / nom_fichier
        
        with open(chemin_principal, 'w', encoding='utf-8') as f:
            f.write(doc_principal)
        
        # 4. Document récapitulatif
        doc_recap = self._generer_recap(nom_fichier, len(fichiers_critiques))
        chemin_recap = self.project_root / f"RECAP_{titre_court}_{self.timestamp}.md"
        
        with open(chemin_recap, 'w', encoding='utf-8') as f:
            f.write(doc_recap)
        
        print(f"✅ Documents créés:")
        print(f"   📄 Principal: {chemin_principal.name}")
        print(f"   📋 Récap: {chemin_recap.name}")
        
        return {
            'principal': str(chemin_principal),
            'recap': str(chemin_recap),
            'fichiers_analyses': len(fichiers_critiques),
            'taille_total': os.path.getsize(chemin_principal) + os.path.getsize(chemin_recap)
        }
    
    def _collecter_code_essentiel(self, fichiers: List[str]) -> Dict[str, str]:
        """Collecte le code essentiel des fichiers spécifiés"""
        code_essentiel = {}
        
        for fichier in fichiers:
            chemin = self.project_root / fichier
            if chemin.exists():
                try:
                    with open(chemin, 'r', encoding='utf-8') as f:
                        contenu = f.read()
                    
                    # Extraction code ou contenu pertinent
                    code_essentiel[fichier] = self._extraire_code_pertinent(contenu, fichier)
                    print(f"  ✅ {fichier} - {len(contenu)} chars")
                    
                except Exception as e:
                    print(f"  ❌ {fichier} - Erreur: {e}")
                    code_essentiel[fichier] = f"# ERREUR LECTURE: {e}"
            else:
                print(f"  ⚠️ {fichier} - Fichier introuvable")
                code_essentiel[fichier] = "# FICHIER INTROUVABLE"
        
        return code_essentiel
    
    def _extraire_code_pertinent(self, contenu: str, fichier: str) -> str:
        """Extrait le code pertinent selon le type de fichier"""
        ext = Path(fichier).suffix.lower()
        
        if ext == '.py':
            # Extraction classes/fonctions principales Python
            return self._extraire_python_essentiel(contenu, fichier)
        elif ext == '.md':
            # Contenu markdown complet
            return contenu
        elif ext in ['.yml', '.yaml']:
            # Configuration YAML
            return contenu
        elif ext == '.json':
            # Configuration JSON
            return contenu
        else:
            # Fichiers autres - première partie
            return contenu[:2000] + "\n\n# ... (contenu tronqué)" if len(contenu) > 2000 else contenu
    
    def _extraire_python_essentiel(self, contenu: str, fichier: str) -> str:
        """Extrait classes/fonctions principales d'un fichier Python"""
        lignes = contenu.split('\n')
        code_essentiel = []
        
        # En-tête avec imports
        for ligne in lignes[:20]:
            if ligne.strip().startswith(('#!', '"""', 'import ', 'from ', 'os.environ')):
                code_essentiel.append(ligne)
        
        # Classes et fonctions principales
        dans_classe = False
        dans_fonction = False
        indent_classe = 0
        indent_fonction = 0
        
        for ligne in lignes:
            stripped = ligne.strip()
            
            # Détection classes
            if stripped.startswith('class '):
                dans_classe = True
                indent_classe = len(ligne) - len(ligne.lstrip())
                code_essentiel.append(ligne)
            
            # Détection fonctions/méthodes
            elif stripped.startswith('def '):
                dans_fonction = True
                indent_fonction = len(ligne) - len(ligne.lstrip())
                code_essentiel.append(ligne)
            
            # Contenu dans classe/fonction
            elif dans_classe or dans_fonction:
                ligne_indent = len(ligne) - len(ligne.lstrip())
                
                # Fin de classe/fonction
                if stripped and ligne_indent <= (indent_classe if dans_classe else indent_fonction):
                    if dans_fonction:
                        dans_fonction = False
                    if dans_classe and ligne_indent <= indent_classe:
                        dans_classe = False
                
                # Docstring, commentaires, ou première ligne de code
                if stripped.startswith(('"""', '#', 'try:', 'if ', 'return ', 'raise ')) or not stripped:
                    code_essentiel.append(ligne)
                elif len([l for l in code_essentiel if l.strip()]) < 150:  # Limite lignes
                    code_essentiel.append(ligne)
        
        resultat = '\n'.join(code_essentiel)
        if len(resultat) > 5000:  # Limite taille
            resultat = resultat[:5000] + "\n\n    # ... (code tronqué pour lisibilité)"
        
        return resultat
    
    def _generer_document_principal(self, probleme: str, code_essentiel: Dict[str, str], 
                                  contexte: str, urgence: str) -> str:
        """Génère le document principal .md avec code agrégé"""
        
        doc = f"""# 🆘 **AIDE EXTERNE - {probleme.upper()}**

**Date** : {datetime.now().strftime("%d %B %Y - %H:%M")}  
**Problème** : {probleme}  
**Urgence** : **{urgence}**  
**SuperWhisper V6** - Phase 4 STT  

---

## 🎯 **CONTEXTE**

{contexte if contexte else "Problème critique nécessitant aide externe spécialisée."}

---

## 🔧 **CODE ESSENTIEL ACTUEL**

"""
        
        # Ajout code essentiel par fichier
        for i, (fichier, code) in enumerate(code_essentiel.items(), 1):
            section_titre = self._generer_titre_section(fichier)
            doc += f"""
### **{i}. {section_titre}**

```python
# {fichier}
{code}
```
"""
        
        doc += """
---

## 🔍 **PROBLÈME IDENTIFIÉ**

### **Zones Critiques**
1. **Architecture/Pipeline** : Analyse du flow de données
2. **Performance** : Goulots d'étranglement identifiés  
3. **Configuration** : Paramètres optimaux manquants
4. **Intégration** : Problèmes de coordination modules

---

## 🆘 **AIDE DEMANDÉE**

### **Solution Complète Attendue**
- **Code fonctionnel immédiatement opérationnel**
- **Configuration optimale pour environnement**
- **Documentation intégration**
- **Plan résolution étape par étape**

### **Contraintes Techniques**
- **GPU** : RTX 3090 24GB exclusif (CUDA:1)
- **OS** : Windows 10 PowerShell 7
- **Python** : 3.12 avec dépendances existantes
- **Performance** : Maintenir niveau actuel

---

**🚨 RÉPONSE EXHAUSTIVE DEMANDÉE AVEC CODE COMPLET !**
"""
        
        return doc
    
    def _generer_titre_section(self, fichier: str) -> str:
        """Génère un titre de section approprié pour le fichier"""
        nom = Path(fichier).stem
        
        if 'unified_stt' in nom:
            return "UnifiedSTTManager - Architecture Principale"
        elif 'backend' in nom:
            return f"Backend {nom.replace('_', ' ').title()}"
        elif 'vad' in nom:
            return "VAD Manager - Voice Activity Detection"
        elif 'validation' in nom:
            return "Script Validation - Point d'Échec"
        elif 'config' in nom:
            return "Configuration Système"
        else:
            return nom.replace('_', ' ').title()
    
    def _generer_recap(self, nom_fichier: str, nb_fichiers: int) -> str:
        """Génère le document récapitulatif"""
        
        return f"""# 📋 **RÉCAPITULATIF AIDE EXTERNE - SUPERWHISPER V6**

**Timestamp** : {self.timestamp}  
**Document principal** : `{nom_fichier}`  
**Fichiers analysés** : {nb_fichiers}  

---

## ✅ **LIVRABLE CRÉÉ**

### **Document Principal**
- **Fichier** : `{nom_fichier}`
- **Format** : Markdown (.md) - Compatible consultants externes
- **Contenu** : Code essentiel agrégé + contexte + demande exhaustive
- **Avantages** :
  - ✅ Un seul fichier à envoyer
  - ✅ Code directement lisible
  - ✅ Pas de dépendances fichiers multiples
  - ✅ Compatible email/chat/collaboration

### **Amélioration vs Package ZIP**
- ❌ **Ancien** : 71 fichiers ZIP (244KB) - Trop lourd
- ✅ **Nouveau** : 1 fichier .md (<50KB) - Optimal
- ✅ Code essentiel extrait et agrégé
- ✅ Lecture directe sans décompression

---

## 🎯 **UTILISATION**

1. **Envoi** : Transmettre `{nom_fichier}` aux consultants
2. **Lecture** : Markdown natif dans tous outils
3. **Réponse** : Code solution directement intégrable
4. **Suivi** : Document unique pour référence

---

**🚀 AIDE EXTERNE OPTIMISÉE - PRÊTE À ENVOYER !**
"""

def main():
    """Interface CLI pour le générateur d'aide externe"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Générateur d'aide externe SuperWhisper V6")
    parser.add_argument("--probleme", required=True, help="Description du problème")
    parser.add_argument("--fichiers", required=True, nargs='+', help="Fichiers critiques à inclure")
    parser.add_argument("--contexte", default="", help="Contexte supplémentaire")
    parser.add_argument("--urgence", default="NORMALE", choices=['NORMALE', 'ÉLEVÉE', 'CRITIQUE'])
    parser.add_argument("--titre", default="AIDE_EXTERNE", help="Titre court pour nommage")
    
    args = parser.parse_args()
    
    generator = GenerateurAideExterne()
    resultat = generator.creer_aide_externe(
        probleme=args.probleme,
        fichiers_critiques=args.fichiers,
        contexte=args.contexte,
        urgence=args.urgence,
        titre_court=args.titre
    )
    
    print(f"\n🎉 Aide externe générée avec succès !")
    print(f"📄 Fichier principal : {resultat['principal']}")
    print(f"📊 Taille totale : {resultat['taille_total']} bytes")

if __name__ == "__main__":
    main() 