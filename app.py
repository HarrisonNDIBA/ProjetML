import streamlit as st
import pandas as pd
import re
import joblib
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 8
})



# ---------------------------------------------------
# CONFIG PAGE
# ---------------------------------------------------
st.set_page_config(
    page_title="SNCF RH – Analyse des candidatures",
    layout="wide",
    page_icon="🚄"
)

# ---------------------------------------------------
# STYLE GLOBAL PRO + DASHBOARD (AJOUT UNIQUEMENT)
# ---------------------------------------------------
st.markdown("""
<style>

/* ===== BACKGROUND GLOBAL ===== */
.stApp {
    background-color: #0E1A2B;
}

/* ===== TEXTE GLOBAL ===== */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif;
    color: #E6ECF2;
}

/* ===== TITRES ===== */
.main-title {
    font-size: 2.6rem;
    font-weight: 900;
    color: #FFFFFF;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #B7C7DA;
    margin-bottom: 2.5rem;
}

/* ===== DASHBOARD ===== */
.dashboard {
    background: #101F36;
    border-radius: 18px;
    padding: 1.8rem;
    margin-bottom: 3rem;
    box-shadow: 0 10px 28px rgba(0,0,0,0.35);
}

.dashboard-title {
    font-size: 1.3rem;
    font-weight: 900;
    color: #FFFFFF;
    margin-bottom: 1.2rem;
}

.kpi {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

.kpi-value {
    font-size: 2rem;
    font-weight: 900;
    color: #1B4079;
}

.kpi-label {
    font-size: 0.9rem;
    font-weight: 700;
    color: #2F3A4A;
}

/* ===== OFFRE ===== */
.offer-box {
    background: linear-gradient(135deg, #F8FBFF, #EEF4F8);
    border-left: 6px solid #1B4079;
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 2.8rem;
    color: #1A1A1A;
}

/* ===== OFFRE RH PREMIUM ===== */
.offer-title {
    font-size: 1.8rem;
    font-weight: 900;
    color: #1B4079;
    margin-bottom: 0.8rem;
}

.offer-subtitle {
    font-size: 1rem;
    font-weight: 700;
    color: #4A5D73;
    margin-bottom: 1.4rem;
}

.offer-section {
    margin-top: 1.2rem;
}

.offer-section h4 {
    font-size: 1.05rem;
    font-weight: 800;
    color: #1B4079;
    margin-bottom: 0.4rem;
}

.offer-section ul {
    padding-left: 1.2rem;
    margin: 0.3rem 0;
}

.offer-section li {
    font-size: 0.95rem;
    color: #2F3A4A;
    margin-bottom: 0.4rem;
    line-height: 1.4;
}


/* ===== BOUTON IA ===== */
div.stButton > button {
    background-color: #FFFFFF;
    color: #1B4079;
    font-weight: 800;
    border-radius: 14px;
    padding: 0.9rem 1.6rem;
    margin: 1.8rem 0 3.2rem 0;
}

/* ===== CARTES CV ===== */
.cv-card {
    background-color: #FFFFFF;
    border-radius: 18px;
    padding: 2.2rem;
    margin-bottom: 3.5rem;
    border-left: 7px solid #1B4079;
}

.cv-card * {
    color: #1A1A1A;
}

.cv-header h3 {
    color: #1B4079 !important;
    font-weight: 900;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 900;
    color: #1B4079 !important;
    border-bottom: 2px solid #D9E3F0;
    margin: 1.8rem 0 0.9rem;
}

.info-row {
    display: flex;
    padding: 0.6rem 0;
}

.info-label {
    min-width: 190px;
    font-weight: 800;
    color: #1B4079 !important;
}

.info-row div:last-child {
    color: #2F3A4A !important;
    font-weight: 600;
}

.photo-box {
    width: 120px;
    height: 150px;
    background-color: #F2F5FA;
    border: 2px dashed #9FB3C8;
    border-radius: 12px;
    color: #1B4079 !important;
    display: flex;
    align-items: center;
    justify-content: center;
}

.skill-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.skill-badge {
    background-color: #E6EEF7;
    color: #1B4079 !important;
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
}

.actions-box {
    background-color: #F4F7FB;
    border-radius: 14px;
    padding: 1.6rem;
    margin-top: 2.2rem;
}

.actions-title {
    font-weight: 900;
    color: #1B4079 !important;
}

.confidential-box {
    background-color: #FFF8E6;
    border-left: 5px solid #E0A800;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-top: 1.6rem;
    font-size: 0.95rem;
    color: #4A3B00 !important;
    font-weight: 700;
}

/* ===== TAG IA (PRÉDICTION MODÈLE – MISE EN ÉVIDENCE) ===== */
.ai-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;

    background: linear-gradient(135deg, #2ECC71, #27AE60);
    color: #FFFFFF;

    font-weight: 900;
    padding: 6px 16px;
    border-radius: 18px;

    font-size: 0.8rem;
    margin-top: 6px;

    box-shadow: 0 4px 12px rgba(39,174,96,0.45);
}


</style>

<style>
/* ===== TAG STATUT RH (RESTAURATION EXACTE) ===== */
.badge-not {
    background-color: #E3EAF3;
    color: #1B4079;
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 800;
}

.badge-processed {
    background-color: #D6F0E4;
    color: #1E7F4D;
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 800;
}
</style>
<style>
/* ===== IA RESULTATS – VERSION PRO & SOBRE ===== */

.ai-highlight {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background-color: #F2F6FB;
    border: 1px solid #D6E0ED;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 800;
    color: #1B4079 !important;
    margin: 10px 0 14px 0;
}

.ai-highlight::before {
    content: "IA";
    background-color: #1B4079;
    color: #FFFFFF;
    font-size: 0.65rem;
    font-weight: 900;
    padding: 2px 8px;
    border-radius: 12px;
}

.ai-highlight b {
    font-weight: 900;
    color: #0F2F5F !important;
}

/* Version secondaire (bas de carte) */
.ai-result {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 800;
    color: #1B4079 !important;
    margin-top: 10px;
}
</style>


""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITRE
# ---------------------------------------------------
st.markdown("<div class='main-title'>Automatisation RH</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyse intelligente des candidatures Data</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# DATA
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data/dataset_cv_clean.xlsx"
MODEL_PATH = BASE_DIR / "models/knn_classification_secteur.pkl"
SCALER_PATH = BASE_DIR / "models/scaler_classification_secteur.pkl"
SECTEUR_MAPPING_PATH = BASE_DIR / "models/secteur_mapping.pkl"
KNN_CONFIG_PATH = BASE_DIR / "models/knn_classification_secteur_config.pkl"



KMEANS_MODEL_PATH = BASE_DIR / "models/kmeans_clustering_model.pkl"
KMEANS_SCALER_PATH = BASE_DIR / "models/kmeans_clustering_scaler.pkl"
KMEANS_FEATURES_PATH = BASE_DIR / "models/kmeans_clustering_features.pkl"


for p in [
    DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    KNN_CONFIG_PATH,
    SECTEUR_MAPPING_PATH,
    KMEANS_MODEL_PATH,
    KMEANS_SCALER_PATH,
    KMEANS_FEATURES_PATH,
]:
    if not p.exists():
        st.error(f"Fichier manquant : {p}")
        st.stop()


@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    knn_config = joblib.load(KNN_CONFIG_PATH)
    FEATURES = knn_config["features"]

    secteur_mapping = joblib.load(SECTEUR_MAPPING_PATH)

    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    kmeans_scaler = joblib.load(KMEANS_SCALER_PATH)
    KMEANS_FEATURES = joblib.load(KMEANS_FEATURES_PATH)

    return model, scaler, FEATURES, secteur_mapping, kmeans_model, kmeans_scaler, KMEANS_FEATURES








df = load_data()
model, scaler, FEATURES, secteur_mapping, kmeans_model, kmeans_scaler, KMEANS_FEATURES = load_models()




st.session_state.setdefault(
    "profil_status",
    {r["ID"]: "Not Processed" for _, r in df.iterrows()}
)
st.session_state.setdefault("decision", {})
st.session_state.setdefault("ai_results", {})
st.session_state.setdefault("page", 1)



# ---------------------------------------------------
# INTERPRÉTATION MÉTIER DES CLUSTERS
# ---------------------------------------------------
CLUSTER_LOGIC = {
    1: {"label": "⭐ Profil Data solide", "priority": "Haute", "color": "#2E8B57"},
    7: {"label": "🎯 Profil confirmé", "priority": "Haute", "color": "#2E8B57"},
    6: {"label": "🎓 Académique / Junior", "priority": "Moyenne", "color": "#1B4079"},
    0: {"label": "🌱 Débutant / Soft skills", "priority": "Faible", "color": "#6C7A89"},
    3: {"label": "❌ Profil faible", "priority": "Très faible", "color": "#B22222"},
    5: {"label": "🚨 CV incomplet", "priority": "Exclusion", "color": "#8B0000"},
    2: {"label": "⚠️ Cas isolé", "priority": "Manuel", "color": "#DAA520"},
    4: {"label": "⚠️ Verbeux / peu qualifié", "priority": "Manuel", "color": "#DAA520"},
}



# ---------------------------------------------------
# MÉTRIQUES DASHBOARD (NÉCESSAIRES AUX GRAPHIQUES)
# ---------------------------------------------------
total_candidats = len(df)

not_processed = sum(
    1 for s in st.session_state.profil_status.values() if s == "Not Processed"
)
processed = total_candidats - not_processed

retenu = sum(
    1 for v in st.session_state.decision.values()
    if v == "Retenu pour un entretien"
)

rejete = sum(
    1 for v in st.session_state.decision.values()
    if v == "Profil rejeté"
)


# ---------------------------------------------------
# DASHBOARD (NOUVELLE SECTION)
# ---------------------------------------------------
total = len(df)
processed = sum(1 for s in st.session_state.profil_status.values() if s == "Processed")
not_processed = total - processed
retenu = sum(1 for d in st.session_state.decision.values() if d == "Retenu pour un entretien")
rejete = sum(1 for d in st.session_state.decision.values() if d == "Profil rejeté")
ia_done = len(st.session_state.ai_results)

st.markdown("<div class='dashboard'>", unsafe_allow_html=True)
st.markdown("<div class='dashboard-title'>Dashboard RH</div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label in [
    (c1, total, "Candidatures"),
    (c2, not_processed, "À traiter"),
    (c3, processed, "Traitées"),
    (c4, retenu, "Retenues"),
    (c5, rejete, "Rejetées"),
]:
    with col:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# OFFRE (INCHANGÉE)
# ---------------------------------------------------
st.markdown("""
<div class="offer-box">

<div class="offer-title">📌 Data Scientist – Automatisation RH (H/F)</div>
<div class="offer-subtitle">
Direction Innovation & Data · Analyse intelligente des candidatures
</div>

<div class="offer-section">
<h4>Contexte</h4>
<p>
Dans le cadre de la modernisation des processus RH, nous développons une plateforme
d’analyse intelligente des candidatures combinant Machine Learning et expertise métier,
afin d’aider les équipes RH à prendre des décisions plus rapides, plus justes et plus objectives.
</p>
</div>

<div class="offer-section">
<h4>Missions principales</h4>
<ul>
    <li>Analyser et structurer des données issues de CV et candidatures</li>
    <li>Concevoir, entraîner et améliorer des modèles de Machine Learning</li>
    <li>Travailler sur des indicateurs de performance et d’aide à la décision RH</li>
    <li>Collaborer avec les équipes RH et Data sur des outils à impact métier</li>
</ul>
</div>

<div class="offer-section">
<h4>Profil recherché</h4>
<ul>
    <li>Formation en Data Science, Machine Learning ou équivalent</li>
    <li>Maîtrise de Python, pandas, scikit-learn</li>
    <li>Sensibilité aux enjeux RH, éthique et explicabilité des modèles</li>
    <li>Capacité à vulgariser des résultats techniques</li>
</ul>
</div>

</div>
""", unsafe_allow_html=True)



# ---------------------------------------------------
# FEATURE ENGINEERING (INCHANGÉ)
# ---------------------------------------------------
def extract_features(row):
    mots = len(str(row["Mots"]).split()) if pd.notna(row["Mots"]) else 0
    comps = len(str(row["Compétences"]).split("|")) if pd.notna(row["Compétences"]) else 0
    langs = len(str(row["Langues"]).split("|")) if pd.notna(row["Langues"]) else 0
    return {
        "Mots": mots,
        "Compétences": comps,
        "A_Email": int(pd.notna(row["Email"])),
        "A_Telephone": int(pd.notna(row["Téléphone"])),
        "A_Permis": int(str(row["Permis"]).lower() == "oui"),
        "Nb_Langues": langs,
        "Nb_Comp_Tech": comps,
        "Ratio_Comp_Mots": comps / mots if mots else 0,
        "Densite_Competences": comps / max(1, mots),
    }

df_features = pd.DataFrame(df.apply(extract_features, axis=1).tolist())

df_features_clf = df_features[FEATURES]
# --- Alignement sécurisé des features KMeans ---
missing_cols = [c for c in KMEANS_FEATURES if c not in df_features.columns]

for col in missing_cols:
    df_features[col] = 0

df_features_kmeans = df_features[KMEANS_FEATURES]


# ---------------------------------------------------
# APPLICATION DU CLUSTERING KMEANS (UNE FOIS)
# ---------------------------------------------------
X_kmeans = df_features_kmeans.copy()


for col in KMEANS_FEATURES:
    if col not in X_kmeans.columns:
        X_kmeans[col] = 0

X_kmeans = X_kmeans[KMEANS_FEATURES]
X_kmeans_scaled = kmeans_scaler.transform(X_kmeans)

df["Cluster_KMeans"] = kmeans_model.predict(X_kmeans_scaled)

df["Segment_Metier"] = df["Cluster_KMeans"].apply(
    lambda c: CLUSTER_LOGIC.get(c, {}).get("label", "Inconnu")
)

df["Priorite_RH"] = df["Cluster_KMeans"].apply(
    lambda c: CLUSTER_LOGIC.get(c, {}).get("priority", "Standard")
)


# ---------------------------------------------------
# IA GLOBAL
# ---------------------------------------------------
if st.button("🔍 Lancer l’analyse IA des candidatures", use_container_width=True):
    X_scaled = scaler.transform(df_features_clf)
    preds = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)

    for i, cid in enumerate(df["ID"]):
        secteur_label = secteur_mapping.get(preds[i], str(preds[i]))

        st.session_state.ai_results[cid] = {
            "secteur": secteur_label,
            "confiance": float(probas[i].max())
        }


    st.success("Analyse IA réalisée avec succès")
    

# ---------------------------------------------------
# 📊 ANALYSES VISUELLES RH (SOBRE & PRO)
# ---------------------------------------------------

st.markdown("<div class='section-title'>Synthèse visuelle</div>", unsafe_allow_html=True)

with st.expander("📂 Ouvrir la synthèse graphique", expanded=False):

    c1, c2, c3 = st.columns([1.6, 1.2, 1.2])

    PRIMARY = "#1B4079"
    SUCCESS = "#2E8B57"
    DANGER = "#B22222"
    MUTED = "#DCE4EE"

    # =========================
    # 1️⃣ PIPELINE RH (BARRE EMPILÉE)
    # =========================
    with c1:
        fig, ax = plt.subplots(figsize=(4.6, 1.2), dpi=800)

        ax.barh(["Candidatures"], [not_processed], color=MUTED, label="À traiter")
        ax.barh(["Candidatures"], [processed], left=[not_processed], color=PRIMARY, label="Traitées")

        ax.set_xlim(0, total)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Avancement du pipeline RH", fontsize=9, fontweight="bold")

        ax.legend(
            loc="lower right",
            frameon=False,
            fontsize=7
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # =========================
    # 2️⃣ DÉCISION RH (DONUT FIN)
    # =========================
    with c2:
        fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=800)

        decision_counts = pd.Series(st.session_state.decision).value_counts()

        if not decision_counts.empty:
            ax.pie(
                decision_counts.values,
                labels=decision_counts.index,
                colors=[SUCCESS, DANGER],
                startangle=90,
                wedgeprops=dict(width=0.32, edgecolor="white"),
                textprops={"fontsize": 7}
            )
            ax.set_title("Décisions RH", fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Aucune décision", ha="center", va="center", fontsize=8)

        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    # =========================
    # 3️⃣ CONFIANCE IA (JAUGE)
    # =========================
    with c3:
        fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=800)

        if st.session_state.ai_results:
            mean_conf = sum(
                v["confiance"] for v in st.session_state.ai_results.values()
            ) / len(st.session_state.ai_results)

            ax.pie(
                [mean_conf, 1 - mean_conf],
                startangle=180,
                colors=[PRIMARY, MUTED],
                wedgeprops=dict(width=0.3)
            )

            ax.text(
                0, -0.05,
                f"{mean_conf:.0%}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=PRIMARY
            )

            ax.set_title("Confiance IA moyenne", fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "IA non lancée", ha="center", va="center", fontsize=8)

        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------
# FILTRE RH – PRIORITÉ MÉTIER
# ---------------------------------------------------
priority_filter = st.selectbox(
    "🎯 Filtrer par priorité RH",
    ["Toutes", "Haute", "Moyenne", "Faible", "Très faible", "Exclusion", "Manuel"]
)

if st.session_state.get("last_filter") != priority_filter:
    st.session_state.page = 1
    st.session_state.last_filter = priority_filter


df_view = df.copy()

if priority_filter != "Toutes":
    df_view = df_view[df_view["Priorite_RH"] == priority_filter]

   

# ---------------------------------------------------
# PAGINATION
# ---------------------------------------------------
PER_PAGE = 10
total_pages = max(1, math.ceil(len(df_view) / PER_PAGE))
start = (st.session_state.page - 1) * PER_PAGE
df_page = df_view.iloc[start:start + PER_PAGE]


# ---------------------------------------------------
# OUTILS CONFIDENTIELS
# ---------------------------------------------------
def clean_text(v):
    return "Non trouvé" if pd.isna(v) or not str(v).strip() else v

def extract_name_from_email(email):
    if pd.isna(email) or "@" not in email:
        return "Nom non disponible"
    local = email.split("@")[0]
    local = re.sub(r"\d+", "", local)
    parts = re.split(r"[._\-]", local)
    parts = [p.capitalize() for p in parts if p]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return parts[0]

def skill_badges(text):
    if pd.isna(text): return
    html = "<div class='skill-badges'>"
    for s in re.split(r"[|,;]", str(text)):
        html += f"<span class='skill-badge'>{s.strip().capitalize()}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------------------------------
# AFFICHAGE CANDIDATS
# ---------------------------------------------------
for _, c in df_page.iterrows():
    cid = c["ID"]
    status = st.session_state.profil_status[cid]

    st.markdown(f"""
    <div class="cv-card">
        <div class="cv-header">
            <div class="cv-left">
                <div class="avatar">👤</div>
                <h3>Candidat n°{cid}</h3>
            </div>
            <span class="{'badge-processed' if status=='Processed' else 'badge-not'}">{status}</span>
        </div>
    """, unsafe_allow_html=True)
    
    # ---------------------------------------------------
    # TAG SEGMENT MÉTIER (KMEANS)
    # ---------------------------------------------------
    cluster_info = CLUSTER_LOGIC.get(c["Cluster_KMeans"], {})

    st.markdown(
        f"""
        <div style="
            background:{cluster_info.get('color', '#DDD')};
            color:white;
            padding:6px 12px;
            border-radius:20px;
            display:inline-block;
            font-size:0.75rem;
            font-weight:700;
            margin-bottom:10px;
        ">
            {cluster_info.get('label', 'Cluster inconnu')} · Priorité {cluster_info.get('priority')}
        </div>
        """,
        unsafe_allow_html=True
    )


    if cid in st.session_state.ai_results:
        r = st.session_state.ai_results[cid]
        st.markdown(
            f"<div class='ai-highlight'>Analyse IA · <b>{r['secteur']}</b> · {r['confiance']:.0%}</div>",
            unsafe_allow_html=True
        )

    col_info, col_photo = st.columns([4,1])
    with col_photo:
        if status == "Processed" and pd.notna(c.get("Photo")):
            st.image(c.get("Photo"), width=120)
        else:
            st.markdown("<div class='photo-box'>PHOTO</div>", unsafe_allow_html=True)




    with col_info:
        st.markdown("<div class='section-title'>Informations générales</div>", unsafe_allow_html=True)
        for lab, key in [("CV","CV"),("Téléphone","Téléphone"),("Code postal","Code_Postal"),("Permis","Permis")]:
            st.markdown(
                f"<div class='info-row'><div class='info-label'>{lab}</div><div>{clean_text(c.get(key))}</div></div>",
                unsafe_allow_html=True
            )

        if status == "Processed":
            email = clean_text(c.get("Email"))
            nom = extract_name_from_email(email)
            st.markdown(
                f"""
                <div class='confidential-box'>
                    <b>Email :</b> {email}<br>
                    <b>Nom :</b> {nom}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='confidential-box'>Informations confidentielles accessibles après validation RH.</div>",
                unsafe_allow_html=True
            )

        st.markdown("<div class='section-title'>Profil</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-row'><div class='info-label'>Diplôme</div><div>{clean_text(c.get('Diplôme'))}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-row'><div class='info-label'>Langues</div><div>{clean_text(c.get('Langues'))}</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Compétences</div>", unsafe_allow_html=True)
        skill_badges(c.get("Liste_Compétences"))

        st.markdown("<div class='actions-box'><div class='actions-title'>Actions RH</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1.4,1.6])
        with col1:
            choix = st.selectbox(
                "Statut",
                ["Aucune sélection","Retenu pour un entretien","Profil rejeté"],
                key=f"select_{cid}",
                disabled=(status=="Processed")
            )

            # ✅ SEULE MODIFICATION : bouton sous le selectbox
            if st.button(f"Terminer la candidature n°{cid}", key=f"btn_{cid}", disabled=(status=="Processed")):
                if choix == "Aucune sélection":
                    st.warning("⚠️ Veuillez sélectionner un statut avant de terminer.")
                else:
                    st.session_state.profil_status[cid] = "Processed"
                    st.session_state.decision[cid] = choix
                    st.rerun()

        with col2:
            pass

        if cid in st.session_state.ai_results:
            r = st.session_state.ai_results[cid]
            st.markdown(
                f"<div class='ai-result'>IA · {r['secteur']} · {r['confiance']:.0%}</div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# NAVIGATION
# ---------------------------------------------------
c1,c2,c3 = st.columns([1,2,1])
with c1:
    if st.button("⬅️ Précédent") and st.session_state.page > 1:
        st.session_state.page -= 1
        st.rerun()
with c2:
    st.markdown(f"<div style='text-align:center;'>Page {st.session_state.page} / {total_pages}</div>", unsafe_allow_html=True)
with c3:
    if st.button("➡️ Suivant") and st.session_state.page < total_pages:
        st.session_state.page += 1
        st.rerun()
