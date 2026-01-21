# 🏢 Modèle de Classification par Secteur d'Activité - KNN

## 📋 Description
Modèle KNN pour attribuer automatiquement un secteur d'activité à des CV.

## 🎯 Objectif
Classifier les CV en 3 secteurs d'activité :
- **Informatique/Tech**
- **Marketing/Communication**
- **Autre**

## 🏆 Performances
- **Accuracy**: 0.8182 (81.82%)
- **F1-Score**: 0.7403
- **Gain vs baseline**: Acc=+12.50% | F1=+13.10%

## ⚙️ Configuration
- **Algorithme**: K-Nearest Neighbors
- **k (voisins)**: 5
- **Scaling**: StandardScaler
- **Features**: 9 variables
- **Classes**: 3 secteurs d'activité

## 📂 Fichiers
1. `knn_classification_secteur.pkl` - Modèle entraîné
2. `scaler_classification_secteur.pkl` - Scaler pour normalisation (si applicable)
3. `knn_classification_secteur_config.pkl` - Configuration complète

## 🚀 Utilisation

```python
import pickle

# Charger le modèle
with open('models/knn_classification_secteur.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger le scaler (si nécessaire)
with open('models/scaler_classification_secteur.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Charger la config
with open('models/knn_classification_secteur_config.pkl', 'rb') as f:
    config = pickle.load(f)

# Prédire le secteur pour un nouveau CV
X_new = [...]  # Features extraites du CV
X_new_scaled = scaler.transform([X_new]) if scaler else [X_new]
secteur_predit = model.predict(X_new_scaled)
print(f"Secteur d'activité prédit : {secteur_predit[0]}")

# Obtenir les probabilités
probas = model.predict_proba(X_new_scaled)
for secteur, proba in zip(config['classes'], probas[0]):
    print(f"  - {secteur}: {proba*100:.1f}%")
```

## 📊 Features utilisées
- Mots
- Compétences
- A_Email
- A_Telephone
- A_Permis
- Nb_Langues
- Nb_Comp_Tech
- Ratio_Comp_Mots
- Densite_Competences

## 📅 Métadonnées
- **Date de création**: 2026-01-08 00:10:33
- **Dataset**: 51 CV
- **Split**: 80/20 (train/test)
