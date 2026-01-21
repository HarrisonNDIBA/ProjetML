import os
import re
import random
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stopwords_fr = stopwords.words('french')

# Charger un modèle IA de similarité sémantique
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# -------------------------
# 1️⃣ Extraction du texte PDF
# -------------------------
def extract_text(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + " "
    except Exception as e:
        print(f"⚠️ Erreur avec {pdf_path}: {e}")
    text = re.sub(r"[\x00-\x1F\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------
# 2️⃣ Extraction simple des compétences
# -------------------------
def detect_skills(text):
    skills = [
        "python","sql","power bi","tableau","excel","r","java","html","css","javascript",
        "react","node.js","django","flask","aws","azure","figma","ui/ux","machine learning",
        "deep learning","seo","scrum","jira","design thinking"
    ]
    found = [s for s in skills if re.search(rf"\b{s}\b", text.lower())]
    return ", ".join(found).title() if found else "Non spécifié"

def detect_poste(text):
    postes = {
        "data analyst": ["data analyst", "analyste de données"],
        "ingénieur machine learning": ["machine learning", "ia", "intelligence artificielle"],
        "chef de projet it": ["chef de projet", "project manager"],
        "ux designer": ["ux", "ui", "design"],
        "développeur web": ["développeur web", "web developer", "frontend", "backend"],
        "chargé de communication": ["communication", "marketing", "contenu"]
    }
    for poste, mots in postes.items():
        if any(re.search(rf"\b{mot}\b", text.lower()) for mot in mots):
            return poste.title()
    return "Poste non détecté"

def detect_langues(text):
    langues = ["français", "anglais", "espagnol", "allemand", "italien", "arabe"]
    found = [l.title() for l in langues if re.search(rf"\b{l}\b", text.lower())]
    return ", ".join(found) if found else "Non spécifié"

# -------------------------
# 3️⃣ Calcul du score IA
# -------------------------
def compute_ai_score(cv_text, job_text):
    emb_cv = model.encode(cv_text, convert_to_tensor=True)
    emb_job = model.encode(job_text, convert_to_tensor=True)
    score = util.cos_sim(emb_cv, emb_job).item()
    return round(score * 100, 2)

# -------------------------
# 4️⃣ Génération du dataset final
# -------------------------
def generate_dataset(cv_folder, job_description, output_file="cv_metadata_list.csv"):
    data = []
    id_counter = 1

    for file in os.listdir(cv_folder):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(cv_folder, file)
        text = extract_text(path)

        nom = file.replace(".pdf", "").replace("_", " ").title()
        age = random.randint(25, 45)
        poste = detect_poste(text)
        competences = detect_skills(text)
        experience = random.randint(1, 15)
        langues = detect_langues(text)
        lien = f"https://example.com/cv/{file}"
        score_ia = compute_ai_score(text, job_description)

        data.append({
            "ID": id_counter,
            "Nom": nom,
            "Âge": age,
            "Poste": poste,
            "Compétences": competences,
            "Expérience (ans)": experience,
            "Langues": langues,
            "Lien vers le CV": lien,
            "Score IA": score_ia
        })
        id_counter += 1

    df = pd.DataFrame(data)
    df = df.sort_values(by="Score IA", ascending=False)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ Dataset IA généré : {output_file}")
    print(df.head(10))

# -------------------------
# 5️⃣ MAIN
# -------------------------
if __name__ == "__main__":
    cv_folder = r"data\CVs"  # ton dossier de CV
    job_description = """
    Nous recherchons un Data Analyst maîtrisant Python, SQL, Power BI et Excel.
    Capable de réaliser des analyses de données, dashboards et rapports, 
    avec une expérience en machine learning et cloud.
    """
    generate_dataset(cv_folder, job_description)
