"""
Module d'extraction de CV PDF
Extrait automatiquement les informations depuis des CV PDF
"""

import os
import re
import PyPDF2
import pandas as pd

# ============================================
# COMPÉTENCES (TECHNIQUES + GÉNÉRALES)
# ============================================
COMPETENCES_TECHNIQUES = [
    # Compétences Tech/Data
    'python', 'sql', 'r', 'excel', 'power bi', 'powerbi', 'tableau', 'javascript',
    'java', 'c++', 'c#', 'machine learning', 'deep learning', 'nlp', 'tensorflow',
    'pytorch', 'scikit-learn', 'scikit', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'spark', 'hadoop', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'k8s',
    'git', 'github', 'mongodb', 'postgresql', 'mysql', 'nosql', 'etl', 'bi',
    'data mining', 'statistics', 'statistique', 'regression', 'classification', 
    'clustering', 'neural networks', 'cnn', 'rnn', 'lstm', 'transformers', 'bert', 
    'gpt', 'flask', 'django', 'fastapi', 'react', 'angular', 'vue', 'node.js', 
    'nodejs', 'html', 'css', 'api', 'rest', 'graphql', 'jupyter', 'notebook',
    'linux', 'bash', 'shell', 'agile', 'scrum', 'keras', 'opencv', 'scipy',
    'airflow', 'kafka', 'redis', 'elasticsearch', 'jenkins', 'terraform', 'ansible',
    # Compétences Bureautiques/Générales
    'office 365', 'office', 'word', 'powerpoint', 'outlook', 'crm', 'erp',
    'communication', 'gestion', 'vente', 'négociation', 'organisation',
    'service client', 'management', 'leadership', 'anglais', 'français'
]

# ============================================
# FONCTIONS D'EXTRACTION
# ============================================

def extraire_texte_pdf(chemin_pdf):
    """
    Extrait le texte d'un fichier PDF
    
    Args:
        chemin_pdf (str): Chemin vers le fichier PDF
        
    Returns:
        str: Texte extrait du PDF
    """
    try:
        texte_complet = ""
        with open(chemin_pdf, 'rb') as fichier:
            lecteur = PyPDF2.PdfReader(fichier)
            for page in lecteur.pages:
                texte = page.extract_text()
                if texte:
                    texte_complet += texte + " "
        return texte_complet.strip()
    except Exception as e:
        print(f"Erreur extraction PDF {chemin_pdf}: {e}")
        return ""


def nettoyer_texte(texte):
    """Nettoie et normalise le texte"""
    texte = texte.lower()
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()


def compter_mots(texte):
    """Compte le nombre de mots dans le texte"""
    return len(texte.split())


def extraire_competences(texte):
    """
    Identifie les compétences techniques et générales
    
    Args:
        texte (str): Texte du CV
        
    Returns:
        list: Liste des compétences trouvées
    """
    texte_clean = nettoyer_texte(texte)
    competences_trouvees = []
    
    for comp in COMPETENCES_TECHNIQUES:
        pattern = r'\b' + re.escape(comp.lower()) + r'\b'
        if re.search(pattern, texte_clean):
            competences_trouvees.append(comp)
    
    return list(set(competences_trouvees))


def extraire_code_postal(texte):
    """Extrait le code postal français (5 chiffres)"""
    pattern = r'\b(\d{5})\b'
    matches = re.findall(pattern, texte)
    
    if matches:
        codes_valides = [code for code in matches 
                        if code[0] != '0' or code[:2] in ['01', '02', '03', '04', '05', '06', '07', '08', '09']]
        if codes_valides:
            return codes_valides[0]
    
    return 'Non trouvé'


def extraire_email(texte):
    """Extrait l'adresse email"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(pattern, texte)
    
    if matches:
        return matches[0]
    
    return 'Non trouvé'


def extraire_telephone(texte):
    """Extrait le numéro de téléphone français"""
    patterns = [
        r'\+33\s*[1-9](?:\s*\d{2}){4}',
        r'0[1-9](?:\s*\d{2}){4}',
        r'\+33[1-9]\d{8}',
        r'0[1-9]\d{8}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, texte)
        if matches:
            tel = matches[0].replace(' ', '')
            return tel
    
    return 'Non trouvé'


def extraire_permis(texte):
    """Extrait le type de permis de conduire"""
    texte_clean = nettoyer_texte(texte)
    permis_trouves = []
    
    patterns_permis = [
        r'permis\s+([a-z])\b',
        r'permis\s+de\s+conduire\s+([a-z])',
        r'\bpermis\s+([a-z])\b'
    ]
    
    for pattern in patterns_permis:
        matches = re.findall(pattern, texte_clean)
        for match in matches:
            permis_trouves.append('Permis ' + match.upper())
    
    if permis_trouves:
        return '; '.join(list(set(permis_trouves)))
    
    return 'Non trouvé'


def extraire_diplome(texte):
    """Extrait le dernier diplôme obtenu"""
    texte_clean = nettoyer_texte(texte)
    
    diplomes = [
        ('doctorat', 'Doctorat'),
        ('phd', 'Doctorat'),
        ('these', 'Doctorat'),
        ('master', 'Master'),
        ('mba', 'MBA'),
        ('bac+5', 'Master'),
        ('bac +5', 'Master'),
        ('ingenieur', 'Ingénieur'),
        ('licence', 'Licence'),
        ('bac+3', 'Licence'),
        ('bac +3', 'Licence'),
        ('bachelor', 'Bachelor'),
        ('dut', 'DUT'),
        ('bts', 'BTS'),
        ('bac+2', 'BTS/DUT'),
        ('bac +2', 'BTS/DUT'),
        ('baccalaureat', 'Baccalauréat'),
        ('bac scientifique', 'Bac Scientifique'),
        ('bac', 'Baccalauréat')
    ]
    
    for keyword, diplome_nom in diplomes:
        if re.search(r'\b' + keyword + r'\b', texte_clean):
            return diplome_nom
    
    return 'Non trouvé'


def extraire_langues(texte):
    """Extrait les langues parlées"""
    texte_clean = nettoyer_texte(texte)
    
    langues_dict = {
        'français': ['français', 'francais', 'french', 'langue maternelle'],
        'anglais': ['anglais', 'english'],
        'espagnol': ['espagnol', 'spanish', 'español'],
        'allemand': ['allemand', 'german', 'deutsch'],
        'italien': ['italien', 'italian', 'italiano'],
        'portugais': ['portugais', 'portuguese', 'português'],
        'chinois': ['chinois', 'chinese', 'mandarin'],
        'arabe': ['arabe', 'arabic'],
        'russe': ['russe', 'russian']
    }
    
    langues_trouvees = set()
    
    for langue, variations in langues_dict.items():
        for variation in variations:
            if re.search(r'\b' + variation + r'\b', texte_clean):
                langues_trouvees.add(langue.capitalize())
                break
    
    if langues_trouvees:
        return '; '.join(sorted(langues_trouvees))
    
    return 'Non trouvé'


def extraire_secteur_activite(texte):
    """
    Identifie le secteur d'activité dominant dans le CV
    
    Args:
        texte (str): Texte du CV
        
    Returns:
        str: Secteur d'activité identifié
    """
    texte_clean = nettoyer_texte(texte)
    
    # Dictionnaire des secteurs avec leurs mots-clés associés
    secteurs = {
        'Informatique/Tech': [
            'développeur', 'developer', 'programmeur', 'software', 'web', 'data scientist',
            'data analyst', 'devops', 'cloud', 'cybersécurité', 'it', 'informatique',
            'fullstack', 'backend', 'frontend', 'mobile', 'application', 'logiciel',
            'système', 'réseau', 'intelligence artificielle', 'machine learning', 'tech'
        ],
        'Finance/Banque': [
            'banque', 'finance', 'comptable', 'audit', 'contrôle de gestion', 
            'analyste financier', 'trader', 'investissement', 'crédit', 'assurance',
            'risque', 'compliance', 'fiscalité', 'trésorerie', 'asset management'
        ],
        'Commerce/Vente': [
            'commercial', 'vente', 'business developer', 'account manager', 'vendeur',
            'négociation', 'prospection', 'relation client', 'sales', 'business development',
            'chargé d\'affaires', 'technico-commercial'
        ],
        'Marketing/Communication': [
            'marketing', 'communication', 'digital marketing', 'community manager',
            'brand manager', 'chef de produit', 'publicité', 'content manager',
            'seo', 'sem', 'social media', 'relations publiques', 'événementiel'
        ],
        'Ressources Humaines': [
            'ressources humaines', 'rh', 'recrutement', 'talent acquisition',
            'chargé de recrutement', 'gestionnaire rh', 'formation', 'paie',
            'relations sociales', 'hr', 'people'
        ],
        'Santé/Médical': [
            'médecin', 'infirmier', 'pharmacien', 'santé', 'médical', 'hôpital',
            'clinique', 'soins', 'patient', 'paramédical', 'kinésithérapeute',
            'dentiste', 'aide-soignant', 'sage-femme'
        ],
        'Industrie/Production': [
            'production', 'usine', 'fabrication', 'industriel', 'maintenance',
            'qualité', 'process', 'supply chain', 'logistique', 'manufacturing',
            'opérateur', 'technicien', 'ingénieur production'
        ],
        'BTP/Construction': [
            'btp', 'construction', 'bâtiment', 'travaux publics', 'génie civil',
            'architecte', 'conducteur de travaux', 'chantier', 'maçon', 'électricien',
            'plombier', 'menuisier'
        ],
        'Éducation/Formation': [
            'enseignant', 'professeur', 'formateur', 'éducation', 'formation',
            'pédagogie', 'école', 'université', 'enseignement', 'teacher',
            'maître', 'instituteur', 'éducateur'
        ],
        'Juridique/Droit': [
            'avocat', 'juriste', 'droit', 'juridique', 'contentieux', 'contrat',
            'légal', 'notaire', 'huissier', 'legal', 'conseil juridique'
        ],
        'Conseil/Audit': [
            'consultant', 'conseil', 'consulting', 'auditeur', 'advisory',
            'stratégie', 'transformation', 'business analyst', 'analyse'
        ],
        'Transport/Logistique': [
            'transport', 'logistique', 'supply chain', 'chauffeur', 'livreur',
            'entreposage', 'distribution', 'approvisionnement', 'flux'
        ],
        'Hôtellerie/Restauration': [
            'hôtel', 'restauration', 'cuisine', 'chef', 'serveur', 'réception',
            'tourisme', 'hébergement', 'hôtellerie', 'cuisinier', 'barman'
        ],
        'Agriculture/Environnement': [
            'agriculture', 'environnement', 'agricole', 'écologie', 'développement durable',
            'vert', 'énergies renouvelables', 'biodiversité', 'agronomie'
        ]
    }
    
    # Compteur de correspondances par secteur
    scores = {}
    
    for secteur, mots_cles in secteurs.items():
        count = 0
        for mot_cle in mots_cles:
            pattern = r'\b' + re.escape(mot_cle) + r'\b'
            matches = re.findall(pattern, texte_clean)
            count += len(matches)
        
        if count > 0:
            scores[secteur] = count
    
    # Retourner le secteur avec le plus de correspondances
    if scores:
        secteur_dominant = max(scores, key=scores.get)
        return secteur_dominant
    
    return 'Non identifié'


# ============================================
# FONCTION PRINCIPALE
# ============================================

def extraire_cv_depuis_dossier(chemin_dossier, verbose=True):
    """
    Extrait tous les CV d'un dossier et crée un DataFrame
    
    Args:
        chemin_dossier (str): Chemin vers le dossier contenant les CV PDF
        verbose (bool): Afficher les messages de progression
        
    Returns:
        pd.DataFrame: DataFrame contenant toutes les informations extraites
    """
    
    if verbose:
        print("🚀 Extraction des CV PDF...")
    
    if not os.path.exists(chemin_dossier):
        print(f"❌ Dossier introuvable: {chemin_dossier}")
        return None
    
    fichiers_pdf = [f for f in os.listdir(chemin_dossier) if f.lower().endswith('.pdf')]
    
    if not fichiers_pdf:
        print(f"❌ Aucun PDF trouvé dans {chemin_dossier}")
        return None
    
    if verbose:
        print(f"✅ {len(fichiers_pdf)} CV trouvés\n")
    
    donnees_cv = []
    
    for idx, fichier in enumerate(fichiers_pdf, 1):
        chemin_complet = os.path.join(chemin_dossier, fichier)
        
        texte_brut = extraire_texte_pdf(chemin_complet)
        
        if not texte_brut or len(texte_brut) < 100:
            donnees_cv.append({
                'ID': idx,
                'CV': fichier,
                'Email': 'NA',
                'Téléphone': 'NA',
                'Code_Postal': 'NA',
                'Permis': 'NA',
                'Diplôme': 'NA',
                'Langues': 'NA',
                'Secteur_Activité': 'NA',
                'Mots': 'NA',
                'Compétences': 'NA',
                'Liste_Compétences': 'NA'
            })
            continue
        
        texte_clean = nettoyer_texte(texte_brut)
        nb_mots = compter_mots(texte_clean)
        competences = extraire_competences(texte_brut)
        nb_competences = len(competences)
        
        donnees_cv.append({
            'ID': idx,
            'CV': fichier,
            'Email': extraire_email(texte_brut),
            'Téléphone': extraire_telephone(texte_brut),
            'Code_Postal': extraire_code_postal(texte_brut),
            'Permis': extraire_permis(texte_brut),
            'Diplôme': extraire_diplome(texte_brut),
            'Langues': extraire_langues(texte_brut),
            'Secteur_Activité': extraire_secteur_activite(texte_brut),
            'Mots': nb_mots,
            'Compétences': nb_competences,
            'Liste_Compétences': '; '.join(competences) if competences else 'Aucune'
        })
    
    df = pd.DataFrame(donnees_cv)
    
    if verbose:
        print(f"✅ Extraction terminée!")
        print(f"📊 {len(df)} CV traités\n")
    
    return df


# ============================================
# TEST DU MODULE
# ============================================

if __name__ == "__main__":
    # Test avec le chemin par défaut
    chemin_test = r"C:\Users\Franck Melvine\Documents\Ecole_et_Aprentissage\Ynov Master\Master 1\Cours\Machine_learning\Projet_CV_ML\data\raw\_Projet_CV"
    
    df = extraire_cv_depuis_dossier(chemin_test)
    
    if df is not None:
        print("📋 APERÇU DU DATAFRAME:")
        print("=" * 150)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 250)
        print(df.head(10))
        print(f"\n✅ DataFrame créé avec {len(df)} lignes et {len(df.columns)} colonnes")
        
        # 🆕 SAUVEGARDE AUTOMATIQUE DU FICHIER CSV 🆕
        chemin_sortie = r"C:\Users\Franck Melvine\Documents\Ecole_et_Aprentissage\Ynov Master\Master 1\Cours\Machine_learning\Projet_CV_ML\data\processed\dataset_cv_clean.csv"
        df.to_csv(chemin_sortie, index=False, encoding='utf-8-sig')
        print(f"\n💾 Fichier sauvegardé : {chemin_sortie}")
        print(f"📊 {len(df.columns)} colonnes sauvegardées : {', '.join(df.columns.tolist())}")