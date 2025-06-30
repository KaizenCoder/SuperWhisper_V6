# 🛠️ Portabilité - Scripts Universels

> **Rendez vos scripts exécutables sur n'importe quel système !**

---

## 🎯 **Outil Principal**

### [`make_scripts_portable.py`](make_scripts_portable.py)
**Rend tous les scripts Python exécutables partout - Windows, Linux, Mac**

#### ⚡ **Usage Rapide**
```bash
# Scanner et corriger tous les scripts
python portability/make_scripts_portable.py --scan-all

# Mode test (voir ce qui sera fait)
python portability/make_scripts_portable.py --dry-run

# Scanner un répertoire spécifique
python portability/make_scripts_portable.py --directory /path/to/scripts
```

#### 🔧 **Fonctionnalités**
- ✅ **Détection automatique** des scripts non-portables
- ✅ **Correction paths** Windows → Universal
- ✅ **Ajout shebangs** pour Linux/Mac
- ✅ **Gestion encodage** UTF-8 automatique
- ✅ **Backup automatique** avant modification
- ✅ **Rapport détaillé** des corrections

#### 📊 **Options Avancées**
```bash
# Scanner récursivement
python portability/make_scripts_portable.py --recursive

# Corriger seulement les imports
python portability/make_scripts_portable.py --fix-imports-only

# Voir les problèmes sans corriger
python portability/make_scripts_portable.py --check-only
```

---

## 🎯 **Cas d'Usage Typiques**

### 🚀 **Démarrage Projet**
Vous avez des scripts qui marchent sous Windows mais pas Linux ?
```bash
python portability/make_scripts_portable.py --scan-all
```

### 🔄 **Maintenance Continue**
Vérification régulière de la portabilité :
```bash
# Tous les lundis
python portability/make_scripts_portable.py --check-only --report
```

### 📦 **Préparation Livraison**
Avant de livrer/partager vos scripts :
```bash
python portability/make_scripts_portable.py --scan-all --backup
```

---

## 📈 **Statistiques**
- **Taille** : 17KB (script complet)
- **Langages supportés** : Python 3.6+
- **Systèmes** : Windows, Linux, MacOS
- **Corrections** : ~15 types de problèmes détectés

---

## 🔗 **Liens Utiles**
- [📖 Guide Complet](../README_PORTABILITE_SCRIPTS.md)
- [🎯 Index Principal](../INDEX_OUTILS_COMPLET.md)
- [🚀 Guide Démarrage](../GUIDE_DEMARRAGE_RAPIDE_OUTILS.md)

---

*Portabilité SuperWhisper V6 - Scripts Universels* 🌍 