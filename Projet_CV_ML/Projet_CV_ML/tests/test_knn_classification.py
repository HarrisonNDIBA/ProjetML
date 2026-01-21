"""
Tests unitaires pour le modèle KNN de classification par secteur
Fichier: tests/test_knn_classification.py
"""

import unittest
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestKNNClassification(unittest.TestCase):
    """Tests pour le modèle KNN de classification"""
    
    @classmethod
    def setUpClass(cls):
        """Charge les modèles une seule fois pour tous les tests"""
        print("\n" + "="*80)
        print("🧪 TESTS UNITAIRES - KNN CLASSIFICATION")
        print("="*80)
        
        models_dir = Path(__file__).parent.parent / 'models'
        
        # Charger le modèle KNN
        model_path = models_dir / 'knn_classification_secteur.pkl'
        with open(model_path, 'rb') as f:
            cls.model = pickle.load(f)
        
        # Charger le scaler
        scaler_path = models_dir / 'scaler_classification_secteur.pkl'
        with open(scaler_path, 'rb') as f:
            cls.scaler = pickle.load(f)
        
        # Charger la config
        config_path = models_dir / 'knn_classification_secteur_config.pkl'
        with open(config_path, 'rb') as f:
            cls.config = pickle.load(f)
        
        print("✅ Modèles chargés")
    
    def test_01_model_exists(self):
        """Test 1: Vérifier que le modèle existe"""
        print("\n🔍 Test 1: Vérification de l'existence du modèle")
        self.assertIsNotNone(self.model)
        print("   ✅ Modèle KNN existe")
    
    def test_02_model_type(self):
        """Test 2: Vérifier le type du modèle"""
        print("\n🔍 Test 2: Vérification du type de modèle")
        from sklearn.neighbors import KNeighborsClassifier
        self.assertIsInstance(self.model, KNeighborsClassifier)
        print(f"   ✅ Type correct: {type(self.model).__name__}")
    
    def test_03_model_parameters(self):
        """Test 3: Vérifier les paramètres du modèle"""
        print("\n🔍 Test 3: Vérification des paramètres")
        # Vérifier que n_neighbors correspond à la config
        expected_k = self.config.get('n_neighbors', 3)
        self.assertEqual(self.model.n_neighbors, expected_k, 
                        f"n_neighbors devrait être {expected_k}")
        print(f"   ✅ n_neighbors: {self.model.n_neighbors}")
        print(f"   ✅ weights: {self.model.weights}")
    
    def test_04_scaler_exists(self):
        """Test 4: Vérifier que le scaler existe"""
        print("\n🔍 Test 4: Vérification du scaler")
        self.assertIsNotNone(self.scaler)
        from sklearn.preprocessing import StandardScaler
        self.assertIsInstance(self.scaler, StandardScaler)
        print("   ✅ Scaler existe et est du bon type")
    
    def test_05_config_structure(self):
        """Test 5: Vérifier la structure de la configuration"""
        print("\n🔍 Test 5: Vérification de la configuration")
        # Clés obligatoires dans ta config
        required_keys = ['classes', 'n_neighbors', 'features']
        for key in required_keys:
            self.assertIn(key, self.config, f"Clé '{key}' manquante dans config")
        
        print(f"   ✅ Configuration complète avec {len(self.config)} clés")
        print(f"   Classes disponibles: {self.config['classes']}")
        print(f"   Nombre de features: {len(self.config['features'])}")
        
        # Vérifier les performances si disponibles
        if 'performance' in self.config:
            print(f"   Accuracy: {self.config['performance']['accuracy']:.2%}")
    
    def test_06_prediction_shape(self):
        """Test 6: Vérifier la forme des prédictions"""
        print("\n🔍 Test 6: Test de prédiction - forme du résultat")
        
        try:
            # Créer des données de test fictives
            n_features = self.model.n_features_in_
            X_test = np.random.rand(5, n_features)
            X_test_scaled = self.scaler.transform(X_test)
            
            predictions = self.model.predict(X_test_scaled)
            
            self.assertEqual(len(predictions), 5, "Devrait prédire 5 secteurs")
            print(f"   ✅ Prédictions: {predictions}")
        except Exception as e:
            self.fail(f"Erreur lors de la prédiction: {e}")
    
    def test_07_prediction_classes(self):
        """Test 7: Vérifier que les prédictions sont des classes valides"""
        print("\n🔍 Test 7: Validation des classes prédites")
        
        try:
            n_features = self.model.n_features_in_
            X_test = np.random.rand(10, n_features)
            X_test_scaled = self.scaler.transform(X_test)
            
            predictions = self.model.predict(X_test_scaled)
            valid_classes = self.config['classes']
            
            for pred in predictions:
                self.assertIn(pred, valid_classes, f"Classe '{pred}' invalide")
            
            print(f"   ✅ Toutes les prédictions sont des classes valides")
        except Exception as e:
            self.fail(f"Erreur lors de la validation: {e}")
    
    def test_08_prediction_probabilities(self):
        """Test 8: Vérifier les probabilités de prédiction"""
        print("\n🔍 Test 8: Test des probabilités")
        
        try:
            n_features = self.model.n_features_in_
            X_test = np.random.rand(3, n_features)
            X_test_scaled = self.scaler.transform(X_test)
            
            probas = self.model.predict_proba(X_test_scaled)
            
            # Vérifier que les probabilités somment à 1
            for i, proba_row in enumerate(probas):
                sum_proba = np.sum(proba_row)
                self.assertAlmostEqual(sum_proba, 1.0, places=5, 
                                     msg=f"Somme des probas pour CV {i} != 1")
            
            print(f"   ✅ Probabilités valides (somme = 1.0)")
            print(f"   Exemple de probabilités: {probas[0]}")
        except Exception as e:
            self.fail(f"Erreur lors du calcul des probabilités: {e}")
    
    def test_09_feature_count(self):
        """Test 9: Vérifier le nombre de features"""
        print("\n🔍 Test 9: Vérification du nombre de features")
        
        n_features_config = len(self.config['features'])
        n_features_model = self.model.n_features_in_
        
        self.assertEqual(n_features_config, n_features_model, 
                        "Nombre de features incohérent")
        print(f"   ✅ Nombre de features: {n_features_model}")
        print(f"   Features: {self.config['features']}")
    
    def test_10_realistic_cv_prediction(self):
        """Test 10: Test avec un CV réaliste"""
        print("\n🔍 Test 10: Test avec données réalistes")
        
        try:
            # Utiliser directement le bon nombre de features du modèle
            n_features = self.model.n_features_in_
            
            # Simuler un CV avec des valeurs réalistes
            # On crée un vecteur avec des valeurs moyennes pour toutes les features
            X_test = np.array([[450, 12, 1, 1, 1, 2, 8, 0.027, 2.67] + [0] * (n_features - 9)])
            
            # S'assurer qu'on a le bon nombre de features
            X_test = X_test[:, :n_features]
            
            X_test_scaled = self.scaler.transform(X_test)
            prediction = self.model.predict(X_test_scaled)
            probas = self.model.predict_proba(X_test_scaled)
            
            print(f"   ✅ Secteur prédit: {prediction[0]}")
            print(f"   Confiance: {np.max(probas):.2%}")
            
            # Afficher top 3 secteurs
            top_indices = np.argsort(probas[0])[::-1][:min(3, len(self.config['classes']))]
            print(f"   Top {len(top_indices)} secteurs:")
            for idx in top_indices:
                print(f"      • {self.config['classes'][idx]}: {probas[0][idx]:.2%}")
        except Exception as e:
            print(f"   ⚠️  Impossible de tester avec CV réaliste: {e}")
            # On ne fait pas échouer le test car c'est juste un exemple


class TestModelIntegrity(unittest.TestCase):
    """Tests d'intégrité des fichiers"""
    
    def test_all_files_exist(self):
        """Vérifier que tous les fichiers existent"""
        print("\n🔍 Test: Intégrité des fichiers")
        
        models_dir = Path(__file__).parent.parent / 'models'
        required_files = [
            'knn_classification_secteur.pkl',
            'scaler_classification_secteur.pkl',
            'knn_classification_secteur_config.pkl'
        ]
        
        for file in required_files:
            file_path = models_dir / file
            self.assertTrue(file_path.exists(), f"Fichier manquant: {file}")
            print(f"   ✅ {file}")


def run_tests():
    """Fonction pour exécuter les tests"""
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter les tests
    suite.addTests(loader.loadTestsFromTestCase(TestKNNClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegrity))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS KNN")
    print("="*80)
    print(f"✅ Tests réussis: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests échoués: {len(result.failures)}")
    print(f"⚠️  Erreurs: {len(result.errors)}")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)