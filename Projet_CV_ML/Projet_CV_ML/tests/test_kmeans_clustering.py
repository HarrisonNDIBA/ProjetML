"""
Tests unitaires pour le modèle KMeans de clustering
Fichier: tests/test_kmeans_clustering.py
"""

import unittest
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestKMeansClustering(unittest.TestCase):
    """Tests pour le modèle KMeans de clustering"""
    
    @classmethod
    def setUpClass(cls):
        """Charge les modèles une seule fois pour tous les tests"""
        print("\n" + "="*80)
        print("🧪 TESTS UNITAIRES - KMEANS CLUSTERING")
        print("="*80)
        
        models_dir = Path(__file__).parent.parent / 'models'
        
        # Charger le modèle KMeans
        model_path = models_dir / 'kmeans_clustering_model.pkl'
        with open(model_path, 'rb') as f:
            cls.model = pickle.load(f)
        
        # Charger le scaler
        scaler_path = models_dir / 'kmeans_clustering_scaler.pkl'
        with open(scaler_path, 'rb') as f:
            cls.scaler = pickle.load(f)
        
        # Charger la config
        config_path = models_dir / 'kmeans_clustering_config.pkl'
        with open(config_path, 'rb') as f:
            cls.config = pickle.load(f)
        
        # Charger les features
        features_path = models_dir / 'kmeans_clustering_features.pkl'
        with open(features_path, 'rb') as f:
            cls.features = pickle.load(f)
        
        print("✅ Modèles chargés")
    
    def test_01_model_exists(self):
        """Test 1: Vérifier que le modèle existe"""
        print("\n🔍 Test 1: Vérification de l'existence du modèle")
        self.assertIsNotNone(self.model)
        print("   ✅ Modèle KMeans existe")
    
    def test_02_model_type(self):
        """Test 2: Vérifier le type du modèle"""
        print("\n🔍 Test 2: Vérification du type de modèle")
        from sklearn.cluster import KMeans
        self.assertIsInstance(self.model, KMeans)
        print(f"   ✅ Type correct: {type(self.model).__name__}")
    
    def test_03_model_parameters(self):
        """Test 3: Vérifier les paramètres du modèle"""
        print("\n🔍 Test 3: Vérification des paramètres")
        self.assertEqual(self.model.n_clusters, 8, "n_clusters devrait être 8")
        self.assertEqual(self.model.init, 'k-means++', "init devrait être 'k-means++'")
        self.assertEqual(self.model.n_init, 20, "n_init devrait être 20")
        self.assertEqual(self.model.max_iter, 500, "max_iter devrait être 500")
        print(f"   ✅ n_clusters: {self.model.n_clusters}")
        print(f"   ✅ init: {self.model.init}")
        print(f"   ✅ n_init: {self.model.n_init}")
        print(f"   ✅ max_iter: {self.model.max_iter}")
    
    def test_04_cluster_centers_exist(self):
        """Test 4: Vérifier que les centres de clusters existent"""
        print("\n🔍 Test 4: Vérification des centres de clusters")
        self.assertTrue(hasattr(self.model, 'cluster_centers_'))
        self.assertEqual(len(self.model.cluster_centers_), 8, 
                        "Devrait avoir 8 centres de clusters")
        print(f"   ✅ {len(self.model.cluster_centers_)} centres de clusters")
    
    def test_05_config_structure(self):
        """Test 5: Vérifier la structure de la configuration"""
        print("\n🔍 Test 5: Vérification de la configuration")
        required_keys = ['n_clusters', 'init', 'n_init', 'max_iter', 
                        'silhouette_score', 'n_features']
        for key in required_keys:
            self.assertIn(key, self.config, f"Clé '{key}' manquante dans config")
        print(f"   ✅ Configuration complète")
        print(f"   Silhouette Score: {self.config['silhouette_score']:.4f}")
    
    def test_06_features_list(self):
        """Test 6: Vérifier la liste des features"""
        print("\n🔍 Test 6: Vérification des features")
        self.assertIsInstance(self.features, list)
        self.assertGreater(len(self.features), 0)
        print(f"   ✅ {len(self.features)} features chargées")
    
    def test_07_prediction_shape(self):
        """Test 7: Vérifier la forme des prédictions"""
        print("\n🔍 Test 7: Test de prédiction - forme du résultat")
        
        n_features = len(self.features)
        X_test = np.random.rand(5, n_features)
        X_test_scaled = self.scaler.transform(X_test)
        
        clusters = self.model.predict(X_test_scaled)
        
        self.assertEqual(len(clusters), 5, "Devrait prédire 5 clusters")
        print(f"   ✅ Clusters prédits: {clusters}")
    
    def test_08_cluster_range(self):
        """Test 8: Vérifier que les clusters sont dans la plage valide"""
        print("\n🔍 Test 8: Validation de la plage des clusters")
        
        n_features = len(self.features)
        X_test = np.random.rand(20, n_features)
        X_test_scaled = self.scaler.transform(X_test)
        
        clusters = self.model.predict(X_test_scaled)
        
        for cluster in clusters:
            self.assertGreaterEqual(cluster, 0, "Cluster < 0")
            self.assertLess(cluster, 8, "Cluster >= 8")
        
        print(f"   ✅ Tous les clusters sont entre 0 et 7")
        print(f"   Distribution: {np.bincount(clusters)}")
    
    def test_09_distance_to_centers(self):
        """Test 9: Vérifier le calcul des distances aux centres"""
        print("\n🔍 Test 9: Test des distances aux centres")
        
        n_features = len(self.features)
        X_test = np.random.rand(5, n_features)
        X_test_scaled = self.scaler.transform(X_test)
        
        distances = self.model.transform(X_test_scaled)
        
        self.assertEqual(distances.shape, (5, 8), 
                        "Distances devrait avoir shape (5, 8)")
        
        # Vérifier que toutes les distances sont positives
        self.assertTrue(np.all(distances >= 0), 
                       "Toutes les distances doivent être >= 0")
        
        print(f"   ✅ Distances calculées correctement")
        print(f"   Distance min: {np.min(distances):.4f}")
        print(f"   Distance max: {np.max(distances):.4f}")
    
    def test_10_realistic_cv_clustering(self):
        """Test 10: Test avec un CV réaliste"""
        print("\n🔍 Test 10: Test avec données réalistes")
        
        # Simuler 3 CVs différents
        cvs_test = [
            # CV Junior
            {'Mots': 250, 'Compétences': 6, 'Ratio_Comp_Mots': 0.024, 
             'Nb_Langues': 1, 'Nb_Comp_Tech': 3},
            # CV Senior
            {'Mots': 600, 'Compétences': 18, 'Ratio_Comp_Mots': 0.030, 
             'Nb_Langues': 3, 'Nb_Comp_Tech': 12},
            # CV Moyen
            {'Mots': 400, 'Compétences': 10, 'Ratio_Comp_Mots': 0.025, 
             'Nb_Langues': 2, 'Nb_Comp_Tech': 6}
        ]
        
        X_test = np.zeros((3, len(self.features)))
        
        for i, cv in enumerate(cvs_test):
            for j, feature in enumerate(self.features):
                if feature in cv:
                    X_test[i, j] = cv[feature]
        
        X_test_scaled = self.scaler.transform(X_test)
        clusters = self.model.predict(X_test_scaled)
        distances = self.model.transform(X_test_scaled)
        
        print(f"   ✅ CV Junior → Cluster {clusters[0]} (dist: {distances[0, clusters[0]]:.2f})")
        print(f"   ✅ CV Senior → Cluster {clusters[1]} (dist: {distances[1, clusters[1]]:.2f})")
        print(f"   ✅ CV Moyen  → Cluster {clusters[2]} (dist: {distances[2, clusters[2]]:.2f})")
    
    def test_11_scaler_consistency(self):
        """Test 11: Vérifier la cohérence du scaler"""
        print("\n🔍 Test 11: Cohérence du scaler")
        
        from sklearn.preprocessing import StandardScaler
        self.assertIsInstance(self.scaler, StandardScaler)
        
        # Vérifier que le scaler a été fitted
        self.assertTrue(hasattr(self.scaler, 'mean_'))
        self.assertTrue(hasattr(self.scaler, 'scale_'))
        
        print(f"   ✅ Scaler correctement fité")
        print(f"   Nombre de features: {len(self.scaler.mean_)}")
    
    def test_12_inertia_check(self):
        """Test 12: Vérifier l'inertie du modèle"""
        print("\n🔍 Test 12: Vérification de l'inertie")
        
        self.assertTrue(hasattr(self.model, 'inertia_'))
        self.assertGreater(self.model.inertia_, 0)
        
        print(f"   ✅ Inertie du modèle: {self.model.inertia_:.4f}")


class TestClusteringIntegrity(unittest.TestCase):
    """Tests d'intégrité des fichiers de clustering"""
    
    def test_all_files_exist(self):
        """Vérifier que tous les fichiers existent"""
        print("\n🔍 Test: Intégrité des fichiers")
        
        models_dir = Path(__file__).parent.parent / 'models'
        required_files = [
            'kmeans_clustering_model.pkl',
            'kmeans_clustering_scaler.pkl',
            'kmeans_clustering_config.pkl',
            'kmeans_clustering_features.pkl'
        ]
        
        for file in required_files:
            file_path = models_dir / file
            self.assertTrue(file_path.exists(), f"Fichier manquant: {file}")
            print(f"   ✅ {file}")


def run_tests():
    """Fonction pour exécuter les tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestKMeansClustering))
    suite.addTests(loader.loadTestsFromTestCase(TestClusteringIntegrity))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS KMEANS")
    print("="*80)
    print(f"✅ Tests réussis: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests échoués: {len(result.failures)}")
    print(f"⚠️  Erreurs: {len(result.errors)}")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)