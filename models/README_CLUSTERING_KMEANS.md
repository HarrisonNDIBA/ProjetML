# 🎯 Modèle de Clustering KMeans - CVs

## 📊 Configuration Optimale

- **Nombre de clusters**: 8
- **Initialisation**: k-means++
- **Nombre d'initialisations**: 20
- **Itérations max**: 500

## 📈 Performance

- **Silhouette Score**: 0.2680
- **Davies-Bouldin Index**: 1.1912
- **Calinski-Harabasz Score**: 8.79

## 📁 Fichiers

- `kmeans_clustering_model.pkl` - Modèle KMeans entraîné
- `kmeans_clustering_scaler.pkl` - StandardScaler pour normalisation
- `kmeans_clustering_features.pkl` - Liste des features utilisées
- `kmeans_clustering_config.pkl` - Configuration complète
- `kmeans_clustering_results.csv` - CVs avec clusters assignés

## 🔧 Utilisation

```python
import pickle
import pandas as pd

# Charger le modèle
with open('models/kmeans_clustering_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger le scaler
with open('models/kmeans_clustering_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prédire pour nouveaux CVs
X_new_scaled = scaler.transform(X_new)
clusters = model.predict(X_new_scaled)
```
