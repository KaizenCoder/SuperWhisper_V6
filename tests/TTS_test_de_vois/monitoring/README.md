# 📊 Monitoring - Surveillance Système en Temps Réel

> **Surveillez les performances de votre système SuperWhisper 24/7 !**

---

## 🎯 **Outils de Monitoring**

### 🔥 **Monitoring Phase 3 Complet**

#### [`monitor_phase3.py`](monitor_phase3.py) - 19KB
**Surveillance complète du système Phase 3 avec alertes**
```bash
python monitoring/monitor_phase3.py
```
- ✅ **GPU RTX 3090** monitoring complet
- ✅ **Pipeline STT/TTS** surveillance temps réel
- ✅ **Performance metrics** détaillées
- ✅ **Alertes automatiques** configurables
- ✅ **Dashboard web** intégré

### 🎬 **Demo Monitoring**

#### [`monitor_phase3_demo.py`](monitor_phase3_demo.py) - 9KB
**Version démonstration pour tests et présentation**
```bash
python monitoring/monitor_phase3_demo.py
```
- ✅ **Interface simplifiée** pour demo
- ✅ **Métriques essentielles** seulement
- ✅ **Mode présentation** intégré
- ✅ **Données simulées** pour tests

---

## 🚀 **Utilisation par Scénario**

### 📊 **Production - Surveillance Continue**
```bash
# Monitoring 24/7 avec alertes
python monitoring/monitor_phase3.py --production --alerts

# Logs détaillés
python monitoring/monitor_phase3.py --verbose --log-file monitor.log

# Dashboard web sur port 8080
python monitoring/monitor_phase3.py --web-dashboard --port 8080
```

### 🎬 **Démonstration & Tests**
```bash
# Demo interactive
python monitoring/monitor_phase3_demo.py --interactive

# Mode présentation
python monitoring/monitor_phase3_demo.py --presentation-mode

# Simulation données
python monitoring/monitor_phase3_demo.py --simulate-load
```

### 🔧 **Développement & Debug**
```bash
# Monitoring développement
python monitoring/monitor_phase3.py --dev-mode --detailed-metrics

# Focus GPU seulement
python monitoring/monitor_phase3.py --gpu-only

# Monitoring spécifique processus
python monitoring/monitor_phase3.py --pid 1234
```

---

## 📈 **Métriques Surveillées**

### 🎮 **GPU RTX 3090**
- **VRAM Usage** : Utilisation mémoire vidéo (%)
- **GPU Load** : Charge processeur graphique (%)
- **Temperature** : Température GPU (°C)
- **Power Draw** : Consommation électrique (W)
- **CUDA Cores** : Utilisation coeurs CUDA (%)
- **Memory Bandwidth** : Bande passante mémoire (GB/s)

### 🖥️ **Système CPU/RAM**
- **CPU Usage** : Utilisation processeur (%)
- **RAM Usage** : Mémoire système utilisée (GB)
- **Disk I/O** : Lecture/écriture disque (MB/s)
- **Network** : Trafic réseau (MB/s)
- **Process Count** : Nombre de processus actifs
- **System Load** : Charge système moyenne

### 🎤 **Pipeline Audio STT/TTS**
- **STT Latency** : Latence reconnaissance vocale (ms)
- **TTS Latency** : Latence synthèse vocale (ms)
- **Audio Quality** : Qualité audio (score 0-100)
- **Processing Time** : Temps traitement pipeline (ms)
- **Queue Length** : Longueur file d'attente
- **Error Rate** : Taux d'erreur pipeline (%)

---

## 🔔 **Système d'Alertes**

### 🚨 **Seuils d'Alerte Critiques**
```bash
# Configuration alertes personnalisées
python monitoring/monitor_phase3.py --alert-gpu-temp 85 --alert-vram 90
```

- **GPU Temp** : >80°C (⚠️) / >85°C (🚨)
- **VRAM Usage** : >85% (⚠️) / >95% (🚨)
- **CPU Usage** : >80% (⚠️) / >90% (🚨)
- **RAM Usage** : >85% (⚠️) / >95% (🚨)
- **STT Latency** : >500ms (⚠️) / >1000ms (🚨)
- **TTS Latency** : >300ms (⚠️) / >600ms (🚨)

### 📧 **Notifications Automatiques**
```bash
# Email notifications
python monitoring/monitor_phase3.py --email-alerts admin@domain.com

# Slack notifications
python monitoring/monitor_phase3.py --slack-webhook https://hooks.slack.com/...

# Windows notifications
python monitoring/monitor_phase3.py --windows-toast
```

---

## 📊 **Dashboard & Rapports**

### 🌐 **Dashboard Web Temps Réel**
```bash
# Lancer dashboard sur localhost:8080
python monitoring/monitor_phase3.py --web-dashboard

# Dashboard externe accessible
python monitoring/monitor_phase3.py --web-dashboard --host 0.0.0.0 --port 8080

# Dashboard avec authentification
python monitoring/monitor_phase3.py --web-dashboard --auth --user admin --pass secret123
```

**Fonctionnalités Dashboard :**
- ✅ **Graphiques temps réel** - Métriques live
- ✅ **Historique** - Données sur 24h/7j/30j
- ✅ **Alertes visuelles** - Notifications dans l'interface
- ✅ **Export données** - CSV, JSON, PDF
- ✅ **Configuration** - Seuils personnalisables

### 📈 **Rapports Automatiques**
```bash
# Rapport quotidien
python monitoring/monitor_phase3.py --daily-report --email-report

# Rapport hebdomadaire
python monitoring/monitor_phase3.py --weekly-report --save-pdf

# Rapport mensuel
python monitoring/monitor_phase3.py --monthly-report --detailed
```

---

## ⚙️ **Configuration Avancée**

### 🎯 **Modes de Fonctionnement**
```bash
# Mode Production (léger, optimisé)
python monitoring/monitor_phase3.py --production

# Mode Development (détaillé, verbose)
python monitoring/monitor_phase3.py --development

# Mode Emergency (alertes immédiates)
python monitoring/monitor_phase3.py --emergency-mode

# Mode Silent (logs uniquement)
python monitoring/monitor_phase3.py --silent --log-only
```

### 📊 **Intervalles de Surveillance**
```bash
# Haute fréquence (1 seconde)
python monitoring/monitor_phase3.py --interval 1

# Fréquence normale (5 secondes)
python monitoring/monitor_phase3.py --interval 5

# Basse fréquence (30 secondes)
python monitoring/monitor_phase3.py --interval 30
```

### 💾 **Stockage & Persistance**
```bash
# Base de données SQLite
python monitoring/monitor_phase3.py --db-sqlite monitor.db

# Export continu CSV
python monitoring/monitor_phase3.py --export-csv --csv-file metrics.csv

# Logs rotatifs
python monitoring/monitor_phase3.py --rotating-logs --max-size 100MB
```

---

## 🔗 **Intégrations**

### 📋 **Avec Autres Outils**
- **Memory** : Alertes fuites → `../memory/`
- **Solutions** : Corrections automatiques → `../solutions/`
- **Testing** : Déclenchement tests → `../testing/`

### 🤖 **Automation**
```bash
# Auto-redémarrage en cas de problème
python monitoring/monitor_phase3.py --auto-restart

# Auto-optimisation GPU
python monitoring/monitor_phase3.py --auto-optimize-gpu

# Auto-nettoyage mémoire
python monitoring/monitor_phase3.py --auto-cleanup
```

---

## 🔗 **Liens Utiles**
- [🧠 Analyse Mémoire](../memory/README.md)
- [💡 Solutions Auto](../solutions/README.md)
- [🧪 Tests Validation](../testing/README.md)
- [🎯 Index Principal](../INDEX_OUTILS_COMPLET.md)

---

*Monitoring SuperWhisper V6 - Surveillance Expert* 📊 