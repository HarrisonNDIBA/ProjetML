# Automatisation RH – Analyse intelligente des candidatures

## Présentation du projet

Ce projet propose une **application web interactive développée avec Streamlit** visant à assister les équipes Ressources Humaines dans l’analyse, la priorisation et la prise de décision sur des candidatures issues de CV.

L’outil combine :

* des techniques de **Machine Learning supervisé** (classification),
* des méthodes de **clustering non supervisé** (segmentation de profils),
* une interface professionnelle orientée **décision métier RH**.

L’objectif principal est de **fluidifier le processus de présélection**, tout en conservant une **lecture humaine, explicable et contrôlée** des résultats.

---

## Fonctionnalités principales

### Analyse automatique des candidatures

* Extraction de variables quantitatives à partir des CV (densité de compétences, langues, longueur du contenu, etc.).
* Classification automatique du **secteur métier probable** à l’aide d’un modèle KNN entraîné.
* Estimation d’un **niveau de confiance** associé à chaque prédiction.

### Segmentation métier des profils

* Application d’un **clustering KMeans** pour regrouper les candidats selon leurs caractéristiques.
* Traduction des clusters en **segments métier interprétables** (profil confirmé, junior, incomplet, à analyser manuellement, etc.).
* Attribution d’une **priorité RH** (haute, moyenne, faible, exclusion, manuel).

### Interface RH interactive

* Tableau de bord global avec indicateurs clés :

  * nombre total de candidatures,
  * candidatures traitées / à traiter,
  * profils retenus ou rejetés,
  * niveau moyen de confiance du modèle.
* Filtres par priorité RH.
* Pagination pour naviguer efficacement dans les candidatures.
* Visualisations synthétiques (avancement du pipeline RH, décisions, confiance IA).

### Gestion manuelle et contrôle humain

* Accès progressif aux informations confidentielles (email, nom).
* Validation RH obligatoire avant décision finale.
* Actions possibles :

  * retenir un candidat pour entretien,
  * rejeter un profil.
* Historisation des décisions dans la session utilisateur.

### Gestion des photos de CV

* Support des chemins Windows ou absolus dans les données sources.
* Normalisation automatique vers des chemins relatifs compatibles avec Streamlit Cloud.
* Affichage conditionnel des photos après validation RH.

---

## Architecture du projet

```text
ProjetML/
│
├── app.py                         # Application Streamlit principale
├── requirements.txt               # Dépendances Python
├── README.md                      # Documentation du projet
│
├── data/
│   └── dataset_cv_clean.xlsx      # Données des candidatures
│
├── models/
│   ├── knn_classification_secteur.pkl
│   ├── scaler_classification_secteur.pkl
│   ├── secteur_mapping.pkl
│   ├── knn_classification_secteur_config.pkl
│   ├── kmeans_clustering_model.pkl
│   ├── kmeans_clustering_scaler.pkl
│   └── kmeans_clustering_features.pkl
│
├── assets/
│   └── photos/
│       └── cv_XX.jpg              # Photos associées aux candidatures
│
└── .gitignore
```

---

## Technologies utilisées

* Python 3.10+
* Streamlit
* Pandas
* Scikit-learn
* Joblib
* Matplotlib

---

## Installation et exécution locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-repo>/ProjetML.git
cd ProjetML
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l’application

```bash
streamlit run app.py
```

L’application sera accessible par défaut à l’adresse :

```
http://localhost:8501
```

---

## Déploiement sur Streamlit Cloud

Pour un déploiement fonctionnel :

* Tous les fichiers requis (`data`, `models`, `assets/photos`) doivent être **présents dans le dépôt GitHub**.
* Les chemins vers les images sont gérés de manière relative pour garantir la compatibilité Linux.
* Le fichier `requirements.txt` doit contenir uniquement des dépendances compatibles avec Streamlit Cloud.

---

## Limites et perspectives d’amélioration

* Les modèles sont actuellement statiques (pas de réentraînement dynamique).
* Les décisions sont conservées uniquement en session (pas de persistance base de données).
* Améliorations possibles :

  * ajout d’une base de données (SQLite, PostgreSQL),
  * traçabilité complète des décisions RH,
  * intégration d’un module d’explicabilité (SHAP),
  * gestion multi-utilisateurs,
  * chargement dynamique de nouveaux CV.

---
