import os
import io
import zipfile
from datetime import datetime
from io import BytesIO
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from fpdf import FPDF
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tempfile, csv, math
from collections import defaultdict
import zipfile

Image.MAX_IMAGE_PIXELS = 1000000000

st.set_page_config(
    page_title="Rhythma, A Heart Failure Readmission Predictor",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)


LIGHT_CSS = """
<style>
    /* Base layout */
    .stApp {
        background-color: #f8fafc;
        color: #1a1a1a;
        font-family: 'Poppins', sans-serif;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #003049 !important;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #00798c, #00b4d8);
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #00b4d8, #00798c);
        color: #ffffff;
        transform: scale(1.03);
    }

    /* Upload box */
    section[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #00b4d8;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    /* Slider */
    [data-testid="stSlider"] .stSlider {
        color: #00798c !important;
    }

    /* Sidebar */        <a href="https://ibb.co/j9ZP4p9B"><img src="https://i.ibb.co/bjRMdfjq/logo.jpg" alt="logo" border="0" /></a>
    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #e0e0e0;
    }

    /* Metrics cards */
    div[data-testid="stMetricValue"] {
        color: #00798c;
    }

    /* Placeholder for Rhythma logo */
    .rhythma-logo {
        width: 220px;
        margin: auto;
        margin-top: 10px;
    }

    /* Placeholder for heart animation */
    .heartbeat {
        text-align: center;
        margin: 20px auto;
        padding: 15px;
        color: #00798c;
        font-weight: 500;
    }
</style>
"""

st.markdown(LIGHT_CSS, unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; padding:10px 20px;">
    <div>
        <h1 style="margin-bottom:0; color:#003049;">Rhythma</h1>
        <p style="color:#00798c; font-size:24px; margin-top:2px;">Where every rhythm tells a story</p>
    </div>
    <div style="display:flex; align-items:center; justify-content:center;">
        <img src="https://i.ibb.co/j9ZP4p9B/logo.jpg"
             alt="Rhythma Logo"
             style="width:160px; height:auto; border-radius:18px; box-shadow:0 4px 8px rgba(0,0,0,0.1);"/>
    </div>
</div>
""", unsafe_allow_html=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT, "assets")
MODELS_DIR = os.path.join(ROOT, "..", "models") if ROOT.endswith("app") else os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FINAL_COLUMNS = [
    "admission_type", "admission_location", "discharge_location", "insurance",
    "hospital_expire_flag", "has_chartevents_data", "gender", "age",
    "lab_mean", "lab_min", "lab_max", "lab_std", "length_of_stay", "severity",
    "lab_range", "lab_ratio", "stay_x_severity", "lab_stability", "readmission_30days"
]

NUMERIC_COLS = [
    "age", "lab_mean", "lab_min", "lab_max", "lab_std", "length_of_stay",
    "lab_range", "lab_ratio", "stay_x_severity", "lab_stability"
]

CATEGORICAL_COLS = [
    "admission_type", "admission_location", "discharge_location", "insurance",
    "hospital_expire_flag", "has_chartevents_data", "gender", "severity"
]

PLOT_SAMPLE_MAX = 5000




def process_readmission_zip(uploaded_bytes):


    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as z:
            z.extractall(tmp_dir)


        files = {f.lower(): os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.lower().endswith(".csv")}
        admissions_path = next((v for k, v in files.items() if "admission" in k), None)
        diagnoses_path = next((v for k, v in files.items() if "diagnos" in k), None)
        patients_path = next((v for k, v in files.items() if "patient" in k), None)
        labs_path = next((v for k, v in files.items() if "labevent" in k), None)

        if not admissions_path or not patients_path:
            raise ValueError("ZIP must contain at least admissions.csv and patients.csv")

        admissions = pd.read_csv(admissions_path, low_memory=False)
        patients = pd.read_csv(patients_path, low_memory=False)
        if diagnoses_path and os.path.exists(diagnoses_path):
            diagnoses = pd.read_csv(diagnoses_path, low_memory=False)
        else:
            diagnoses = pd.DataFrame(columns=["subject_id", "hadm_id"])


        merged_df = admissions.merge(diagnoses, on=["subject_id", "hadm_id"], how="left")


        for col in ["admittime", "dischtime", "dob", "dod"]:
            if col in merged_df.columns:
                merged_df[col] = pd.to_datetime(merged_df[col], errors="coerce")
            elif col in patients.columns:
                patients[col] = pd.to_datetime(patients[col], errors="coerce")

        merged_df = merged_df.merge(
            patients[["subject_id", "gender", "dob", "dod"]].drop_duplicates("subject_id"),
            on="subject_id", how="left"
        )


        merged_df["age"] = np.nan
        mask = merged_df["admittime"].notna() & merged_df["dob"].notna()
        valid = merged_df[mask].index
        merged_df.loc[valid, "age"] = (merged_df.loc[valid, "admittime"] - merged_df.loc[valid, "dob"]).dt.days / 365.25
        merged_df.loc[merged_df["age"] < 0, "age"] = np.nan
        merged_df.loc[merged_df["age"] > 150, "age"] = np.nan


        if "dischtime" in merged_df.columns and "admittime" in merged_df.columns:
            merged_df["length_of_stay"] = (
                (merged_df["dischtime"] - merged_df["admittime"]).dt.total_seconds() / 86400
            ).clip(lower=0)
        else:
            merged_df["length_of_stay"] = 0


        if labs_path and os.path.exists(labs_path):
            print("Aggregating lab events efficiently...")
            lab_means = defaultdict(list)
            with open(labs_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    try:
                        hadm = row.get("hadm_id")
                        if not hadm:
                            continue
                        val = float(row.get("valuenum", "nan"))
                        if not math.isfinite(val) or abs(val) > 1e6:
                            continue
                        lab_means[hadm].append(val)
                    except Exception:
                        continue
                    if i % 2_000_000 == 0 and i > 0:
                        print(f"Processed {i:,} rows...")

            rows = []
            for hadm, vals in lab_means.items():
                n = len(vals)
                if n == 0:
                    continue
                mean_v = sum(vals) / n
                min_v = min(vals)
                max_v = max(vals)
                var = sum((v - mean_v) ** 2 for v in vals) / n if n > 1 else 0
                std_v = math.sqrt(var)
                rows.append((int(hadm), mean_v, min_v, max_v, std_v))

            lab_summary = pd.DataFrame(rows, columns=["hadm_id", "lab_mean", "lab_min", "lab_max", "lab_std"])
            merged_df = merged_df.merge(lab_summary, on="hadm_id", how="left")
        else:
            merged_df["lab_mean"] = merged_df["lab_min"] = merged_df["lab_max"] = merged_df["lab_std"] = np.nan


        merged_df["lab_range"] = merged_df["lab_max"] - merged_df["lab_min"]
        merged_df["lab_ratio"] = merged_df["lab_mean"] / merged_df["lab_std"].replace(0, np.nan)
        merged_df["lab_stability"] = 1 / (1 + merged_df["lab_std"].fillna(0))
        merged_df["stay_x_severity"] = merged_df["length_of_stay"] * pd.to_numeric(merged_df.get("severity", 0), errors="coerce")


        final_cols = [
            "admission_type", "admission_location", "discharge_location", "insurance",
            "hospital_expire_flag", "has_chartevents_data", "gender", "age",
            "lab_mean", "lab_min", "lab_max", "lab_std", "length_of_stay", "severity",
            "lab_range", "lab_ratio", "stay_x_severity", "lab_stability",
            "readmission_30days", "synthetic_flag"
        ]
        for col in final_cols:
            if col not in merged_df.columns:
                merged_df[col] = 0

        merged_df = merged_df[final_cols]
        merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
        merged_df = merged_df.fillna(0)
        print(f"Final processed shape: {merged_df.shape}")
        return merged_df


def safe_read_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.read_csv(io.StringIO(file_bytes.decode("utf-8", errors="replace")))


def extract_zip_to_temp(uploaded_bytes: bytes) -> str:
    tmp_dir = os.path.join(ROOT, "tmp_upload")
    os.makedirs(tmp_dir, exist_ok=True)

    for f in os.listdir(tmp_dir):
        try:
            os.remove(os.path.join(tmp_dir, f))
        except Exception:
            pass
    with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as z:
        z.extractall(tmp_dir)
    return tmp_dir


def enforce_final_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for col in FINAL_COLUMNS:

        if col == "readmission_30days" and col not in df.columns:
            continue
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = 0.0 if col in NUMERIC_COLS else 0

    for c in NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    for c in CATEGORICAL_COLS:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    cols = [c for c in FINAL_COLUMNS if c in out.columns]
    return out[cols]



@st.cache_data(show_spinner=False)
def cached_groupby_sum(df_serialized: bytes, col: str, target: str) -> pd.DataFrame:
    """
    df_serialized: pass df.to_pickle() bytes to avoid hashing issues
    """
    df = pd.read_pickle(BytesIO(df_serialized))
    return df.groupby(col, observed=True)[target].sum().reset_index()


@st.cache_data(show_spinner=False)
def cached_corr_matrix(df_serialized: bytes) -> pd.DataFrame:
    df = pd.read_pickle(BytesIO(df_serialized))
    return df.select_dtypes(include=[np.number]).corr()


def df_to_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    df.to_pickle(bio)
    return bio.getvalue()

def fit_preprocessors(df: pd.DataFrame, prefix="preproc"):
    encoders = {}
    for c in CATEGORICAL_COLS:
        le = LabelEncoder()
        vals = df[c].astype(str).fillna("NA")
        le.fit(vals)
        encoders[c] = le
    scaler = StandardScaler().fit(df[NUMERIC_COLS].fillna(0))
    joblib.dump(encoders, os.path.join(MODELS_DIR, f"{prefix}_encoders.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{prefix}_scaler.pkl"))
    return encoders, scaler


def load_preprocessors(prefix="preproc"):
    enc_path = os.path.join(MODELS_DIR, f"{prefix}_encoders.pkl")
    sca_path = os.path.join(MODELS_DIR, f"{prefix}_scaler.pkl")
    enc = joblib.load(enc_path) if os.path.exists(enc_path) else None
    sca = joblib.load(sca_path) if os.path.exists(sca_path) else None
    return enc, sca


def apply_preprocessors(df: pd.DataFrame, encoders=None, scaler=None, fit_if_missing=True):
    df_proc = df.copy()

    df_proc = df_proc.drop(columns=["readmission_30days", "synthetic_flag"], errors="ignore")

    if encoders is None and fit_if_missing:
        encoders, _ = fit_preprocessors(df_proc)
    if scaler is None and fit_if_missing:
        _, scaler = fit_preprocessors(df_proc)

    for c in CATEGORICAL_COLS:
        if c in df_proc.columns:
            le = encoders.get(c) if (encoders and c in encoders) else LabelEncoder()
            df_proc[c] = le.fit_transform(df_proc[c].astype(str).fillna("NA"))


    for c in NUMERIC_COLS:
        if c not in df_proc.columns:
            df_proc[c] = 0.0
    df_proc[NUMERIC_COLS] = scaler.transform(df_proc[NUMERIC_COLS].fillna(0))
    return df_proc, encoders, scaler


def load_lgb_model():
    model_path = os.path.join(MODELS_DIR, "lightgbm_model.pkl")
    return joblib.load(model_path) if os.path.exists(model_path) else None


def train_and_save_lgb(df: pd.DataFrame):

    train_df = df.copy().drop(columns=["synthetic_flag"], errors="ignore")
    if "readmission_30days" not in train_df.columns:
        raise ValueError("Training dataset must contain 'readmission_30days'.")
    y = train_df["readmission_30days"].astype(int)
    X = train_df.drop(columns=["readmission_30days"])

    X = enforce_final_schema(X).drop(columns=["readmission_30days"], errors="ignore")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    try:
        model.feature_name_ = list(X_train.columns)
    except Exception:
        pass
    joblib.dump(model, os.path.join(MODELS_DIR, "lightgbm_model.pkl"))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc


def sample_for_plot(df: pd.DataFrame, max_rows: int = PLOT_SAMPLE_MAX) -> pd.DataFrame:
    n = len(df)
    if n <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def safe_savefig(fig: plt.Figure, path: str, dpi: int = 120):
    """
    Save figure with constrained size & DPI to avoid enormous images.
    """
    try:
        fig.set_size_inches(8, 4.5)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    except Exception:
        fig.set_size_inches(6, 3)
        fig.savefig(path, dpi=80, bbox_inches="tight", pad_inches=0.1)



def generate_pdf_report(summary_info: dict, image_paths: list, out_path: str):
    def clean_text(s):
        if isinstance(s, str):
            return s.replace("—", "-").replace("–", "-").encode("latin-1", "replace").decode("latin-1")
        return str(s)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, clean_text("Rhythma - Readmission Report"), ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    for k, v in summary_info.items():
        pdf.cell(0, 6, clean_text(f"{k}: {v}"), ln=True)
    pdf.ln(8)

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        try:
            with Image.open(img_path) as im:
                max_px = 2000
                im.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
                tmp_safe = img_path.replace(".png", "_safe.png")
                im.save(tmp_safe, "PNG", optimize=True)
            pdf.image(tmp_safe, w=185)
            pdf.ln(6)
            try:
                os.remove(tmp_safe)
            except Exception:
                pass
        except Exception:
            continue

    pdf.output(out_path)
    return out_path


with st.sidebar:
    st.header("Upload & Run")
    uploaded = st.file_uploader("Upload CSV or ZIP (processed CSV or raw MIMIC ZIP)", type=["csv", "zip"])
    run_button = st.button("Process & Predict")
    st.markdown("---")
    auto_train = st.checkbox("Auto-train LGBM if missing (requires labeled dataset.csv at repo root)", value=True)
    threshold = st.slider("Prediction threshold", min_value=0.01, max_value=0.5, value=0.10, step=0.01)



if uploaded is None:
    st.info("Upload a processed CSV or a ZIP containing raw MIMIC files (admissions/diagnoses/patients/labevents). The app will merge & preprocess automatically.")


    st.markdown("<hr style='margin-top:50px; margin-bottom:20px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#003049; margin-bottom:20px;'>How It Works</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background-color:#f0f9ff; border:1px solid #d0ebf7; padding:25px; border-radius:12px; text-align:center;'>
            <div style='font-size:36px; color:#00798c;'>🩺</div>
            <h4 style='color:#003049;'>Data Collection</h4>
            <p style='color:#1a1a1a;'>Collects patient data including vitals, labs, and demographics for predictive modeling.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background-color:#f0f9ff; border:1px solid #d0ebf7; padding:25px; border-radius:12px; text-align:center;'>
            <div style='font-size:36px; color:#00798c;'>📊</div>
            <h4 style='color:#003049;'>Risk Analysis</h4>
            <p style='color:#1a1a1a;'>Analyzes trends using LightGBM to identify patients most at risk of readmission.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background-color:#f0f9ff; border:1px solid #d0ebf7; padding:25px; border-radius:12px; text-align:center;'>
            <div style='font-size:36px; color:#00798c;'>💡</div>
            <h4 style='color:#003049;'>Actionable Insights</h4>
            <p style='color:#1a1a1a;'>Generates recommendations to help clinicians reduce preventable readmissions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center; color:#003049; margin-top:50px;'>Benefits</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#4b6b74;'>Rhythma empowers healthcare providers with AI-driven insights to reduce readmissions and enhance patient outcomes.</p>", unsafe_allow_html=True)

    benefits = [
        ("Improved Patient Outcomes", "Early identification enables timely interventions and better recovery."),
        ("Reduced Readmission Rates", "Hospitals using Rhythma have seen reduced 30-day readmission trends."),
        ("Cost Efficiency", "Lower readmissions translate into improved operational and financial outcomes."),
        ("Resource Optimization", "Focus resources on high-risk patients and optimize clinical workload."),
        ("Data-Driven Decisions", "Gain transparency into risk patterns and driver variables using SHAP analysis."),
        ("Regulatory Compliance", "Aligns with CMS and healthcare quality improvement standards.")
    ]

    for i in range(0, len(benefits), 3):
        cols = st.columns(3)
        for j, (title, desc) in enumerate(benefits[i:i+3]):
            with cols[j]:
                st.markdown(f"""
                <div style='background-color:#f0f9ff; border:1px solid #d0ebf7; padding:25px; border-radius:12px; margin:10px; min-height:160px;'>
                    <h4 style='color:#003049;'>{title}</h4>
                    <p style='color:#1a1a1a;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center; color:#003049; margin-top:50px;'>Key Features</h2>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("""
        <div style='padding:15px;'>
            <div style='margin-bottom:25px;'>
                <h4 style='color:#003049;'> Predictive Analytics Dashboard</h4>
                <p style='color:#1a1a1a;'>Visualize patient risk factors, probabilities, and outcome distributions interactively.</p>
            </div>
            <div style='margin-bottom:25px;'>
                <h4 style='color:#003049;'>Automated Follow-up Scheduling</h4>
                <p style='color:#1a1a1a;'>Suggests follow-up timing based on predicted risk and historical data trends.</p>
            </div>
            <div style='margin-bottom:25px;'>
                <h4 style='color:#003049;'> Care Plan Generator</h4>
                <p style='color:#1a1a1a;'>Creates personalized recommendations for each patient based on their profile.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:

        st.components.v1.html(
            """
            <div style="background-color:#f0f9ff; border:1px solid #d0ebf7; padding:25px; border-radius:12px; font-family: Poppins, sans-serif;">
                <h4 style="color:#003049; margin-top:0; margin-bottom:8px;">Patient Risk Assessment</h4>

                <p style="margin:6px 0 4px 0;">Readmission Risk</p>
                <div style="height:10px; background:#e5eef2; border-radius:4px; overflow:hidden;">
                    <div style="width:78%; height:10px; background:#00798c; border-radius:4px;"></div>
                </div>
                <p style="color:#00798c; font-size:13px; margin:6px 0 12px 0;">High (78%)</p>

                <p style="margin:6px 0 4px 0;">Medication Adherence</p>
                <div style="height:10px; background:#e5eef2; border-radius:4px; overflow:hidden;">
                    <div style="width:45%; height:10px; background:#00b4d8; border-radius:4px;"></div>
                </div>
                <p style="color:#00b4d8; font-size:13px; margin:6px 0 12px 0;">Medium (45%)</p>

                <p style="margin:6px 0 4px 0;">Follow-up Attendance</p>
                <div style="height:10px; background:#e5eef2; border-radius:4px; overflow:hidden;">
                    <div style="width:85%; height:10px; background:#72efdd; border-radius:4px;"></div>
                </div>
                <p style="color:#0096c7; font-size:13px; margin:6px 0 12px 0;">Good (85%)</p>
            </div>
            """,
            height=320,
            scrolling=True,
        )


    st.markdown("<hr style='margin-top:40px;'>", unsafe_allow_html=True)
    st.caption("Rhythma — where every rhythm tells a story")

    st.stop()



try:
    file_bytes = uploaded.read()
    is_zip = uploaded.name.lower().endswith(".zip")
    if is_zip:
        st.info("Extracting and processing ZIP file...")
        df_pre = process_readmission_zip(file_bytes)
    else:
        df_pre = safe_read_csv(file_bytes)
except Exception as e:
    st.error(f"Failed to load uploaded file: {e}")
    st.stop()


if "synthetic_flag" in df_pre.columns:
    df_pre = df_pre.drop(columns=["synthetic_flag"], errors="ignore")


df_pre = enforce_final_schema(df_pre)

st.success("Preprocessing completed (schema enforced).")
st.dataframe(df_pre.head())



encoders, scaler = load_preprocessors()
if encoders is None or scaler is None:
    encoders, scaler = fit_preprocessors(df_pre)
    st.info("Fitted & saved encoders/scaler on uploaded data.")



df_proc, encoders, scaler = apply_preprocessors(df_pre, encoders, scaler)
st.write(f"✅ Preprocessed shape: {df_proc.shape}")

model = load_lgb_model()
if model is None:
    if auto_train:
        st.info("No LightGBM model found. Attempting to auto-train using 'dataset.csv' at repo root (must include readmission_30days).")
        dataset_fp = os.path.join(os.path.dirname(ROOT), "dataset.csv")
        if not os.path.exists(dataset_fp):
            st.error("Auto-train requested but 'dataset.csv' not found at project root.")
            st.stop()
        train_df = pd.read_csv(dataset_fp)
        if "synthetic_flag" in train_df.columns:
            train_df = train_df.drop(columns=["synthetic_flag"], errors="ignore")
        train_df = enforce_final_schema(train_df)
        # train
        try:
            model, auc = train_and_save_lgb(train_df)
            st.success(f"Trained model (AUC={auc:.3f}) and saved to models/")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()
        model = load_lgb_model()
    else:
        st.error("No model found and auto-train disabled.")
        st.stop()
else:
    st.success("Loaded LightGBM model.")



if hasattr(model, "feature_name_"):
    trained_feats = list(model.feature_name_)

    for f in trained_feats:
        if f not in df_proc.columns:
            df_proc[f] = 0

    extras = [c for c in df_proc.columns if c not in trained_feats]
    if extras:
        df_proc = df_proc.drop(columns=extras)

    df_proc = df_proc[trained_feats]


try:
    probs = model.predict_proba(df_proc)[:, 1]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

preds = (probs >= threshold).astype(int)
df_pre["predicted_readmission"] = preds
df_pre["readmission_probability"] = probs

expected_readm = float(df_pre["readmission_probability"].sum())
n_risk = int(df_pre["predicted_readmission"].sum())

st.metric("Expected readmissions (sum of probabilities)", f"{expected_readm:.2f}")
st.metric("Predicted at-risk patients", f"{n_risk} / {len(df_pre)}")

st.markdown("### Predictions (preview)")
st.dataframe(df_pre.head())

csv_bytes = df_pre.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV", data=csv_bytes, file_name="rhythma_predictions.csv", mime="text/csv")



st.markdown("---")
st.markdown("##  Advanced Insights Dashboard")


df_bytes = df_to_bytes(df_pre)


tab1, tab2, tab3 = st.tabs(["Overview", "Risk Segmentation", "Feature Impact"])


with tab1:
    st.subheader("Key Distribution Trends")
    col1, col2 = st.columns(2)


    with col1:
        plot_df = sample_for_plot(df_pre)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.violinplot(x=plot_df["severity"].astype(int), y=plot_df["readmission_probability"], palette="crest", ax=ax1, inner=None,legend=False,hue="severity")
        sns.boxplot(x=plot_df["severity"].astype(int), y=plot_df["readmission_probability"], width=0.18, showcaps=True, boxprops={"facecolor": "white"}, showfliers=False, ax=ax1)
        sns.stripplot(x=plot_df["severity"].astype(int), y=plot_df["readmission_probability"], color="black", alpha=0.35, size=2, jitter=0.25, ax=ax1)
        means = plot_df.groupby("severity", observed=True)["readmission_probability"].mean().reindex(sorted(plot_df["severity"].unique()))
        for i, (sev, m) in enumerate(means.items()):
            ax1.text(i, m + 0.01, f"{m:.3f}", ha="center", color="#003049", fontsize=9)
        ax1.set_title("Readmission Probability by Severity")
        ax1.set_xlabel("Severity Level")
        ax1.set_ylabel("Predicted Readmission Probability")
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        fig1.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)
        st.pyplot(fig1)


    with col2:
        grp = cached_groupby_sum(df_bytes, "age", "readmission_probability")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.lineplot(x="age", y="readmission_probability", data=grp, linewidth=2, ax=ax2)
        sns.scatterplot(x="age", y="readmission_probability", data=grp, alpha=0.6, ax=ax2)
        ax2.set_title("Expected Readmissions by Age")
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Expected Readmissions")
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)

    avg_prob = df_pre["readmission_probability"].mean()
    high_risk_pct = df_pre["predicted_readmission"].mean() * 100
    st.markdown(f"**Summary Insight:** Average predicted readmission probability **{avg_prob:.2%}**, **{high_risk_pct:.1f}%** patients flagged as at-risk (threshold: {threshold:.2f}).")



with tab2:
    st.subheader("Patient Risk Segmentation")

    bins = [0, 0.05, 0.1, 0.2, 1.0]
    labels = ["Low", "Moderate", "High", "Critical"]
    df_pre["risk_tier"] = pd.cut(df_pre["readmission_probability"], bins=bins, labels=labels, include_lowest=True)
    risk_counts = df_pre["risk_tier"].value_counts().reindex(labels).fillna(0).astype(int)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=risk_counts.index, y=risk_counts.values, palette="RdYlBu_r", ax=ax3)
    ax3.set_title("Patient Risk Distribution by Tier")
    ax3.set_xlabel("Risk Tier")
    ax3.set_ylabel("Number of Patients")
    for i, v in enumerate(risk_counts.values):
        ax3.text(i, v + max(1, int(len(df_pre) * 0.01)), str(v), ha="center", color="#003049", fontsize=9)
    fig3.tight_layout()
    st.pyplot(fig3)

    col1, col2 = st.columns(2)
 
    with col2:
        avg_insurance = (
            df_pre.groupby("insurance", observed=True)["readmission_probability"]
            .mean()
            .reset_index()
            .sort_values("readmission_probability", ascending=False)
            .head(10)
        )

        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x="insurance",
            y="readmission_probability",
            data=avg_insurance,
            palette="Blues",
            ax=ax5
        )
        ax5.set_title("Average Readmission Probability by Top 10 Insurance Types", fontsize=12)
        ax5.set_xlabel("Insurance Type")
        ax5.set_ylabel("Readmission Probability")
        ax5.tick_params(axis="x", rotation=45)
        fig5.tight_layout()
        st.pyplot(fig5)

    st.markdown("#### Top 10 At-Risk Patients")
    top_risk = df_pre.nlargest(10, "readmission_probability")[["age", "severity", "readmission_probability"]]
    st.dataframe(top_risk.style.format({"readmission_probability": "{:.3f}"}))



with tab3:
    st.subheader("Feature Impact Analysis")
    st.markdown("#### Feature Correlation with Readmission Probability")

    corr = cached_corr_matrix(df_bytes)
    if "readmission_probability" in corr.columns:

        target_corr = corr["readmission_probability"].abs().sort_values(ascending=False)
        top_feats = target_corr.head(30).index
        corr_small = corr.loc[top_feats, top_feats]

        fig6, ax6 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_small, cmap="vlag", center=0, linewidths=0.2, ax=ax6)
        ax6.set_title("Top Feature Correlation Heatmap (limited to 30)")
        fig6.tight_layout()
        st.pyplot(fig6)
    else:
        st.info("Correlation heatmap not available — 'readmission_probability' missing.")



st.markdown("### Feature importance (LightGBM)")
fi = pd.DataFrame({"feature": df_proc.columns, "importance": model.feature_importances_})
fi = fi.sort_values("importance", ascending=False).head(20)
fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
sns.barplot(x="importance", y="feature", data=fi, ax=ax_fi, palette="mako")
ax_fi.set_title("Top features (LightGBM)")
fig_fi.tight_layout()
st.pyplot(fig_fi)



st.markdown("### Generate PDF report")

def collect_and_save_figs_for_pdf():

    figs_to_include = []

    for candidate in ["fig1", "fig2", "fig3", "fig6", "fig_fi"]:
        obj = locals().get(candidate)
        if isinstance(obj, plt.Figure):
            figs_to_include.append(obj)

    if not figs_to_include:
        figs_to_include.append(plt.gcf())

    saved_paths = []
    for i, fig in enumerate(figs_to_include):
        path = os.path.join(ROOT, f"tmp_plot_{i}.png")
        safe_savefig(fig, path, dpi=120)
        saved_paths.append(path)
    return saved_paths

if st.button("Generate PDF report"):
    with st.spinner("Generating PDF — optimizing images and packaging..."):
        tmp_imgs = collect_and_save_figs_for_pdf()
        summary_info = {
            "Generated on": datetime.utcnow().isoformat(),
            "Total rows": len(df_pre),
            "Predicted at-risk": n_risk,
            "Expected readmissions": f"{expected_readm:.2f}"
        }
        out_pdf = os.path.join(ROOT, f"rhythma_report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf")
        try:
            generate_pdf_report(summary_info, tmp_imgs, out_pdf)
            with open(out_pdf, "rb") as f:
                st.download_button("Download PDF report", data=f, file_name=os.path.basename(out_pdf), mime="application/pdf")
            st.success("PDF generated and ready to download.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
        finally:
            for p in tmp_imgs:
                try:
                    os.remove(p)
                except Exception:
                    pass

st.markdown("---")
st.caption("Rhythma — where every rhythm tells a story")
