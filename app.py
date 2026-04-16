"""
╔══════════════════════════════════════════════════════════╗
║   Supply Chain Intelligence Hub — Streamlit Dashboard    ║
║   Built for: Late Delivery Risk Prediction Project       ║
╚══════════════════════════════════════════════════════════╝

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import io
import os
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── sklearn ───────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_OK = False

try:
    from category_encoders import OneHotEncoder as OHE
    CE_OK = True
except ImportError:
    CE_OK = False

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Supply Chain Intelligence Hub",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 { color: #e2e8f0; font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { color: #94a3b8; font-size: 0.95rem; margin: 0.4rem 0 0; }

.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    margin-bottom: 0.5rem;
}
.kpi-card .kpi-value { font-size: 2rem; font-weight: 700; color: #38bdf8; margin: 0; }
.kpi-card .kpi-label { font-size: 0.78rem; color: #94a3b8; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-card .kpi-sub   { font-size: 0.7rem; color: #64748b; margin-top: 0.2rem; }

.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 4px solid #38bdf8;
    padding-left: 0.75rem;
    margin: 1.2rem 0 0.8rem;
}

.risk-badge-high { background:#ef4444; color:white; padding:3px 12px; border-radius:99px; font-weight:600; font-size:0.8rem; }
.risk-badge-low  { background:#22c55e; color:white; padding:3px 12px; border-radius:99px; font-weight:600; font-size:0.8rem; }

.model-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.model-box h3 { color: #38bdf8; margin: 0 0 0.8rem; }

.pred-result-late { background: linear-gradient(135deg, #7f1d1d, #991b1b); border:1px solid #ef4444; border-radius:12px; padding:1.5rem; text-align:center; }
.pred-result-ok   { background: linear-gradient(135deg, #14532d, #166534); border:1px solid #22c55e; border-radius:12px; padding:1.5rem; text-align:center; }
.pred-result-late h2, .pred-result-ok h2 { color: white; margin: 0; font-size: 1.6rem; }
.pred-result-late p,  .pred-result-ok p  { color: #d1d5db; margin: 0.4rem 0 0; }

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
}

.supplier-good { color: #22c55e; font-weight: 600; }
.supplier-bad  { color: #ef4444; font-weight: 600; }
.supplier-mid  { color: #f59e0b; font-weight: 600; }

div[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR
# ════════════════════════════════════════════════════════════
@st.cache_data
def generate_synthetic_data(n=5000, seed=42):
    """Generate realistic supply chain data matching DataCo statistics."""
    rng = np.random.default_rng(seed)

    markets = ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"]
    market_weights = [0.30, 0.28, 0.23, 0.14, 0.05]
    shipping_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    ship_weights = [0.59, 0.19, 0.15, 0.07]
    payment_types = ["DEBIT", "TRANSFER", "CASH", "PAYMENT"]
    pay_weights = [0.39, 0.27, 0.18, 0.16]
    categories = [
        "Fishing","Cleats","Camping & Hiking","Cardio Equipment","Women's Apparel",
        "Water Sports","Men's Footwear","Indoor/Outdoor Games","Shop By Sport","Computers",
        "Electronics","Team Sports","Accessories","Garden"
    ]
    departments = ["Fan Shop","Apparel","Golf","Outdoors","Footwear","Technology","Disc Shop"]
    order_statuses = ["COMPLETE","PENDING","PROCESSING","CLOSED","SUSPECTED_FRAUD","CANCELED"]
    regions = [
        "Western Europe","Central America","South America","Northern Europe",
        "Oceania","Southern Europe","Southeast Asia","Caribbean","South Asia","West of USA"
    ]
    delivery_statuses_map = {
        0: ["Advance shipping","Shipping on time"],
        1: ["Late delivery"],
    }

    n_late = int(n * 0.549)
    n_ok   = n - n_late

    late_flags = [1] * n_late + [0] * n_ok
    rng.shuffle(late_flags)

    df = pd.DataFrame()
    df["Late_delivery_risk"] = late_flags

    df["Market"]        = rng.choice(markets, n, p=market_weights)
    df["Shipping Mode"] = rng.choice(shipping_modes, n, p=ship_weights)
    df["Type"]          = rng.choice(payment_types, n, p=pay_weights)
    df["Category Name"] = rng.choice(categories, n)
    df["Department Name"] = rng.choice(departments, n)
    df["Order Region"]  = rng.choice(regions, n)
    df["Order Status"]  = rng.choice(order_statuses, n, p=[0.55,0.1,0.1,0.1,0.07,0.08])
    df["Customer Segment"] = rng.choice(["Consumer","Corporate","Home Office"], n, p=[0.52,0.31,0.17])
    df["Customer Country"] = rng.choice(["United States","Mexico","France","Germany","Brazil"], n)

    # Real shipping days: late orders tend to be higher
    days_real = np.where(
        df["Late_delivery_risk"] == 1,
        rng.integers(4, 8, n),
        rng.integers(1, 5, n)
    )
    days_sched = rng.integers(2, 7, n)
    df["Days for shipping (real)"]       = days_real
    df["Days for shipment (scheduled)"]  = days_sched

    df["Sales"]                = rng.uniform(20, 2000, n).round(2)
    df["Sales per customer"]   = rng.uniform(50, 500, n).round(2)
    df["Benefit per order"]    = rng.normal(50, 120, n).round(2)
    df["Order Profit Per Order"] = rng.normal(40, 90, n).round(2)
    df["Order Item Discount"]  = rng.uniform(0, 0.35, n).round(3)
    df["Order Item Quantity"]  = rng.integers(1, 10, n)
    df["Product Price"]        = rng.choice([19.99,29.99,49.99,79.99,99.99,149.99,199.99,299.99,399.99], n)
    df["Order Item Total"]     = (df["Product Price"] * df["Order Item Quantity"]).round(2)
    df["Order Item Profit Ratio"] = rng.uniform(-0.5, 0.5, n).round(3)
    df["Order Item Product Price"] = df["Product Price"]

    # Delivery Status
    def make_delivery_status(row):
        if row["Late_delivery_risk"] == 1:
            return "Late delivery"
        else:
            return rng.choice(["Advance shipping","Shipping on time"], p=[0.56,0.44])
    df["Delivery Status"] = df.apply(make_delivery_status, axis=1)

    # Dates
    base = pd.Timestamp("2017-01-01")
    order_offsets = rng.integers(0, 365*2, n)
    df["order date (DateOrders)"] = [base + pd.Timedelta(days=int(d)) for d in order_offsets]
    df["shipping date (DateOrders)"] = [
        row["order date (DateOrders)"] + pd.Timedelta(days=int(row["Days for shipping (real)"]))
        for _, row in df.iterrows()
    ]

    # Supplier name (synthetic — derived from department+market combo)
    supplier_pool = [
        f"Supplier_{m[:3].upper()}_{d[:4].upper()}"
        for m in markets for d in departments
    ][:20]
    df["Supplier"] = rng.choice(supplier_pool, n)

    return df


# ════════════════════════════════════════════════════════════
#  DATA PREPROCESSING PIPELINE
# ════════════════════════════════════════════════════════════
def preprocess(df_raw):
    df = df_raw.copy()

    # Convert dates if present
    for col in ["shipping date (DateOrders)", "order date (DateOrders)"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "shipping date (DateOrders)" in df.columns and "order date (DateOrders)" in df.columns:
        df["Order_to_Shipment_Time"] = (
            (df["shipping date (DateOrders)"] - df["order date (DateOrders)"])
            .dt.total_seconds() / 3600
        ).fillna(0).astype(int)
        df["ship_day_of_week"] = df["shipping date (DateOrders)"].dt.dayofweek.fillna(0).astype(int)
        df["order_day_of_week"] = df["order date (DateOrders)"].dt.dayofweek.fillna(0).astype(int)
        df["ship_hour"]  = df["shipping date (DateOrders)"].dt.hour.fillna(0).astype(int)
        df["order_hour"] = df["order date (DateOrders)"].dt.hour.fillna(0).astype(int)

    # Drop columns we don't need for modelling
    drop_cols = [
        "Customer Id","Customer Fname","Customer Lname","Customer Email","Customer Password",
        "Customer Street","Customer Zipcode","Customer Phone","Product Description","Product Image",
        "Order Zipcode","Order Customer Id","order id","customer id","Department Id","Product Status",
        "order date (DateOrders)","shipping date (DateOrders)","Customer City","Order City",
        "Order Region","Order State","Order Status","Customer Country","Customer Segment",
        "Market","Delivery Status","Product Name","Category Name","Department Name",
        "Product Card Id","Order Item Id","Order Item Cardprod Id",
        "ship_day_of_week_name","order_day_of_week_name","ship_daypart","order_daypart",
        "Order Country","Supplier","Latitude","Longitude",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    if "Late_delivery_risk" not in df.columns:
        st.error("❌ Dataset must contain a 'Late_delivery_risk' column (0/1).")
        st.stop()

    # Fill missing numerics with mean, categoricals with mode
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # Encode Shipping Mode ordinally if present
    if "Shipping Mode" in df.columns:
        oe = OrdinalEncoder(categories=[["Standard Class","Second Class","First Class","Same Day"]],
                            handle_unknown="use_encoded_value", unknown_value=-1)
        df["Shipping Mode"] = oe.fit_transform(df[["Shipping Mode"]])

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


# ════════════════════════════════════════════════════════════
#  TRAIN MODELS
# ════════════════════════════════════════════════════════════
@st.cache_resource
def train_models(df_raw):
    df = preprocess(df_raw)
    X = df.drop("Late_delivery_risk", axis=1)
    y = df["Late_delivery_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    X_train_s = X_train.copy(); X_test_s = X_test.copy()
    X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_s[num_cols]  = scaler.transform(X_test[num_cols])

    # ── Random Forest ──────────────────────────────
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)

    # ── XGBoost / fallback ─────────────────────────
    if XGBOOST_OK:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=42, n_jobs=-1)
    else:
        xgb = GradientBoostingClassifier(n_estimators=100, random_state=42)

    xgb.fit(X_train_s, y_train)
    y_pred_xgb = xgb.predict(X_test_s)

    # ── Logistic Regression ────────────────────────
    lr = LogisticRegression(solver="liblinear", random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)

    feature_names = X.columns.tolist()
    fi_rf  = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    fi_xgb = pd.Series(xgb.feature_importances_, index=feature_names).sort_values(ascending=False)

    return {
        "rf": rf, "xgb": xgb, "lr": lr,
        "scaler": scaler, "num_cols": num_cols,
        "feature_names": feature_names,
        "X_test": X_test_s, "y_test": y_test,
        "y_pred_rf": y_pred_rf, "y_pred_xgb": y_pred_xgb, "y_pred_lr": y_pred_lr,
        "fi_rf": fi_rf, "fi_xgb": fi_xgb,
    }


# ════════════════════════════════════════════════════════════
#  HELPER CHARTS (matplotlib → streamlit)
# ════════════════════════════════════════════════════════════
DARK_BG  = "#0f172a"
DARK_AX  = "#1e293b"
TEXT_CLR = "#cbd5e1"
ACC_BLUE = "#38bdf8"
ACC_RED  = "#f87171"
ACC_GRN  = "#4ade80"
ACC_AMB  = "#fbbf24"

def style_fig(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
    return fig


def pie_chart(values, labels, title, colors=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    if colors is None:
        colors = [ACC_BLUE, ACC_RED, ACC_GRN, ACC_AMB, "#a78bfa", "#fb923c"]
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct="%1.1f%%", startangle=90,
        colors=colors[:len(values)], pctdistance=0.8,
        wedgeprops={"linewidth": 2, "edgecolor": DARK_BG}
    )
    for at in autotexts:
        at.set(fontsize=8, color="white", fontweight="bold")
    ax.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=7.5, labelcolor=TEXT_CLR,
              facecolor=DARK_AX, edgecolor="#334155")
    ax.set_title(title, color=TEXT_CLR, fontsize=11, fontweight="600", pad=10)
    style_fig(fig)
    return fig


def bar_chart(x, y, title, xlabel="", ylabel="", color=ACC_BLUE, horiz=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    if horiz:
        ax.barh(x, y, color=color, edgecolor=DARK_BG, linewidth=0.5)
        ax.set_xlabel(ylabel, color=TEXT_CLR, fontsize=9)
        ax.set_ylabel(xlabel, color=TEXT_CLR, fontsize=9)
        ax.invert_yaxis()
    else:
        ax.bar(x, y, color=color, edgecolor=DARK_BG, linewidth=0.5)
        ax.set_xlabel(xlabel, color=TEXT_CLR, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_CLR, fontsize=9)
        plt.xticks(rotation=35, ha="right", fontsize=8)
    ax.set_title(title, color=TEXT_CLR, fontsize=11, fontweight="600")
    ax.grid(axis="x" if horiz else "y", color="#334155", linewidth=0.5, linestyle="--")
    style_fig(fig)
    plt.tight_layout()
    return fig


def confusion_matrix_fig(cm, title):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=1, linecolor="#334155",
                annot_kws={"size": 14, "weight": "bold", "color": "white"},
                xticklabels=["On-Time","Late"], yticklabels=["On-Time","Late"])
    ax.set_xlabel("Predicted", color=TEXT_CLR, fontsize=9)
    ax.set_ylabel("Actual", color=TEXT_CLR, fontsize=9)
    ax.set_title(title, color=TEXT_CLR, fontsize=11, fontweight="600")
    style_fig(fig)
    plt.tight_layout()
    return fig


def feature_importance_fig(fi_series, title, n=15, color=ACC_BLUE):
    top = fi_series.head(n)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=color, edgecolor=DARK_BG, linewidth=0.5)
    # gradient colour
    for i, bar in enumerate(bars):
        bar.set_alpha(0.6 + 0.4 * (i / len(bars)))
    ax.set_xlabel("Importance Score", color=TEXT_CLR, fontsize=9)
    ax.set_title(title, color=TEXT_CLR, fontsize=11, fontweight="600")
    ax.grid(axis="x", color="#334155", linewidth=0.5, linestyle="--")
    style_fig(fig)
    plt.tight_layout()
    return fig


def model_comparison_fig(names, accs, precs, recs, f1s):
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - 1.5*w, accs,  w, label="Accuracy",  color=ACC_BLUE,  alpha=0.9)
    b2 = ax.bar(x - 0.5*w, precs, w, label="Precision", color=ACC_GRN,   alpha=0.9)
    b3 = ax.bar(x + 0.5*w, recs,  w, label="Recall",    color=ACC_AMB,   alpha=0.9)
    b4 = ax.bar(x + 1.5*w, f1s,   w, label="F1-Score",  color="#a78bfa", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names, color=TEXT_CLR, fontsize=9)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score", color=TEXT_CLR, fontsize=9)
    ax.set_title("Model Performance Comparison", color=TEXT_CLR, fontsize=11, fontweight="600")
    ax.legend(fontsize=8, labelcolor=TEXT_CLR, facecolor=DARK_AX, edgecolor="#334155")
    ax.grid(axis="y", color="#334155", linewidth=0.5, linestyle="--")
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                    f"{h:.3f}", ha="center", va="bottom", color=TEXT_CLR, fontsize=6.5)
    style_fig(fig)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚚 Supply Chain Hub")
    st.markdown("---")

    st.markdown("### 📂 Data Source")
    use_default = st.radio(
        "Choose data source",
        ["Use built-in demo data", "Upload company data"],
        index=0
    )

    uploaded_file = None
    if use_default == "Upload company data":
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel only",
            type=["csv", "xlsx"],
            help="File must contain supply chain order data with a 'Late_delivery_risk' (0/1) column.",
            accept_multiple_files=False,
        )
        if uploaded_file and not uploaded_file.name.lower().endswith((".csv", ".xlsx")):
            st.error("❌ Only .csv or .xlsx files are accepted.")
            uploaded_file = None

    st.markdown("---")
    st.markdown("### 📌 Navigation")
    st.markdown("""
- 🏠 **Overview** – KPIs & project summary  
- 🔍 **EDA** – Charts & distributions  
- 🤖 **ML Models** – RF & XGBoost outputs  
- 🏭 **Suppliers** – Reliability dashboard  
- 🔮 **Predict** – Single shipment check  
    """)

    st.markdown("---")
    st.caption("Dataset: DataCo Smart Supply Chain\n\nModels: Logistic Regression, Random Forest, XGBoost")


# ════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_uploaded(file_bytes, fname):
    if fname.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
    else:
        return pd.read_excel(io.BytesIO(file_bytes))


if uploaded_file is not None:
    try:
        raw_df = load_uploaded(uploaded_file.read(), uploaded_file.name)
        data_label = f"📤 Uploaded: `{uploaded_file.name}`"
    except Exception as e:
        st.error(f"Error reading file: {e}")
        raw_df = generate_synthetic_data()
        data_label = "📊 Built-in Demo Data (DataCo Supply Chain)"
else:
    raw_df = generate_synthetic_data()
    data_label = "📊 Built-in Demo Data (DataCo Supply Chain)"


# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="main-header">
  <h1>🚚 Supply Chain Intelligence Hub</h1>
  <p>Predicting Late Delivery Risk · Random Forest · XGBoost · Logistic Regression &nbsp;|&nbsp; {data_label}</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "🔍 EDA", "🤖 ML Models", "🏭 Suppliers", "🔮 Predict"
])


# ────────────────────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ────────────────────────────────────────────────────────────
with tab1:
    total_orders  = len(raw_df)
    late_pct      = raw_df["Late_delivery_risk"].mean() * 100 if "Late_delivery_risk" in raw_df.columns else 0
    ontime_pct    = 100 - late_pct

    sales_col = "Sales" if "Sales" in raw_df.columns else None
    total_sales = raw_df[sales_col].sum() if sales_col else 0

    cols = st.columns(4)
    kpis = [
        ("Total Orders",    f"{total_orders:,}",      "Records in dataset"),
        ("Late Delivery %", f"{late_pct:.1f}%",        "Orders at risk"),
        ("On-Time Rate",    f"{ontime_pct:.1f}%",      "Delivery success"),
        ("Total Revenue",   f"${total_sales/1e6:.2f}M","Gross sales"),
    ]
    for col, (label, val, sub) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Project Description
    with st.expander("📖 Project Introduction", expanded=True):
        st.markdown("""
**Predicting Late Delivery Risk in Supply Chain Management**

In today's competitive e-commerce and logistics environment, timely delivery is critical to maintaining
customer satisfaction and supply chain efficiency. Late deliveries negatively impact customer loyalty,
increase operational costs, and may lead to penalties or lost business.

This project leverages the **DataCo Smart Supply Chain Dataset** (180,519 records, 53 columns) to build
machine learning models that accurately predict whether an order is at risk of being delivered late.

**Approach:**
- Extensive Exploratory Data Analysis (EDA) to understand data patterns
- Feature engineering: Order-to-Shipment Time, Day-of-Week, Daypart features
- Outlier treatment using Z-score (normal) and IQR (skewed) methods
- Categorical feature selection using Chi-Square tests; numerical using Pearson correlation
- Models trained: **Logistic Regression**, **Random Forest**, **XGBoost**
- All models achieve ~**97.4% accuracy** on the test set

**Key Insight:** The actual shipping time (`Days for shipping (real)`) and the `Order_to_Shipment_Time`
are the top predictors of late delivery risk — logistics timing matters far more than product type or payment method.
        """)

    # Quick stats
    st.markdown('<div class="section-title">📈 Key Statistics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if "Delivery Status" in raw_df.columns:
            ds = raw_df["Delivery Status"].value_counts()
            st.subheader("Delivery Status Breakdown")
            st.dataframe(
                ds.reset_index().rename(columns={"count":"Count","Delivery Status":"Status"}),
                use_container_width=True, hide_index=True
            )
    with c2:
        if "Shipping Mode" in raw_df.columns:
            sm = raw_df["Shipping Mode"].value_counts()
            st.subheader("Shipping Mode Usage")
            st.dataframe(
                sm.reset_index().rename(columns={"count":"Count","Shipping Mode":"Mode"}),
                use_container_width=True, hide_index=True
            )

    st.markdown('<div class="section-title">🗂 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(10), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.metric("Columns", raw_df.shape[1])
    c4.metric("Rows", f"{raw_df.shape[0]:,}")


# ────────────────────────────────────────────────────────────
#  TAB 2 — EDA
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Row 1: Delivery Status pie + Market bar
    c1, c2 = st.columns(2)

    with c1:
        if "Delivery Status" in raw_df.columns and "Sales" in raw_df.columns:
            ds_sales = raw_df.groupby("Delivery Status")["Sales"].sum()
            colors = {
                "Late delivery": ACC_RED,
                "Advance shipping": ACC_BLUE,
                "Shipping on time": ACC_GRN,
                "Shipping canceled": "#94a3b8",
            }
            clrs = [colors.get(k, ACC_AMB) for k in ds_sales.index]
            fig = pie_chart(ds_sales.values, ds_sales.index, "Sales by Delivery Status", clrs)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption(
                "💡 Late deliveries account for the **largest share** of total sales, "
                "indicating a significant operational issue that directly impacts customer satisfaction."
            )
        else:
            if "Late_delivery_risk" in raw_df.columns:
                vc = raw_df["Late_delivery_risk"].value_counts()
                fig = pie_chart(vc.values, ["On-Time","Late"], "Late Delivery Risk Split",
                                [ACC_GRN, ACC_RED])
                st.pyplot(fig, use_container_width=True)
                plt.close()

    with c2:
        if "Market" in raw_df.columns and "Sales" in raw_df.columns:
            mkt = raw_df.groupby("Market")["Sales"].sum().sort_values(ascending=False)
            mkt_colors = [ACC_BLUE, "#818cf8", ACC_AMB, ACC_GRN, "#fb923c"]
            fig = bar_chart(mkt.index, mkt.values / 1e6,
                            "Sales by Market", "Market", "Sales (Millions $)",
                            color=mkt_colors[:len(mkt)])
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("💡 **Europe** and **LATAM** dominate sales, while Africa has the smallest contribution.")

    st.markdown("---")

    # Row 2: Category + Shipping mode
    c3, c4 = st.columns(2)

    with c3:
        cat_col = "Category Name" if "Category Name" in raw_df.columns else None
        if cat_col and "Sales" in raw_df.columns:
            cat_sales = raw_df.groupby(cat_col)["Sales"].sum().sort_values(ascending=False).head(10)
            fig = bar_chart(cat_sales.index, cat_sales.values / 1e6,
                            "Top 10 Product Categories by Sales",
                            "Category", "Sales (Millions $)", color=ACC_BLUE, horiz=True)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("💡 **Fishing** leads all product categories in total sales revenue.")

    with c4:
        if "Shipping Mode" in raw_df.columns and "Late_delivery_risk" in raw_df.columns:
            sm_lr = raw_df.groupby("Shipping Mode")["Late_delivery_risk"].mean() * 100
            palette = [ACC_RED if v > 60 else ACC_AMB if v > 40 else ACC_GRN for v in sm_lr.values]
            fig = bar_chart(sm_lr.index, sm_lr.values,
                            "Late Delivery Risk % by Shipping Mode",
                            "Shipping Mode", "Late Delivery %", color=palette)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("💡 **First Class** has the highest late delivery rate despite its premium name.")

    st.markdown("---")

    # Row 3: Payment type analysis + Delivery status stacked bar
    c5, c6 = st.columns(2)

    with c5:
        if "Type" in raw_df.columns and "Late_delivery_risk" in raw_df.columns:
            pay = raw_df.groupby("Type")["Late_delivery_risk"].agg(
                late="sum", total="count"
            ).reset_index()
            pay["on_time"] = pay["total"] - pay["late"]
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(pay))
            w = 0.35
            ax.bar(x - w/2, pay["on_time"], w, label="On-Time", color=ACC_GRN, alpha=0.9)
            ax.bar(x + w/2, pay["late"],    w, label="Late",    color=ACC_RED,  alpha=0.9)
            ax.set_xticks(x); ax.set_xticklabels(pay["Type"], color=TEXT_CLR, fontsize=9)
            ax.set_ylabel("Orders", color=TEXT_CLR, fontsize=9)
            ax.set_title("Payment Type vs Late Delivery Risk", color=TEXT_CLR, fontsize=11, fontweight="600")
            ax.legend(fontsize=8, labelcolor=TEXT_CLR, facecolor=DARK_AX, edgecolor="#334155")
            ax.grid(axis="y", color="#334155", linewidth=0.5, linestyle="--")
            style_fig(fig)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("💡 **DEBIT** payments show the highest count of late deliveries.")

    with c6:
        if "Shipping Mode" in raw_df.columns and "Delivery Status" in raw_df.columns:
            try:
                pivot = raw_df.groupby(["Shipping Mode","Delivery Status"]).size().unstack(fill_value=0)
                pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
                fig, ax = plt.subplots(figsize=(6, 4))
                colors_stack = [ACC_BLUE, ACC_RED, "#94a3b8", ACC_GRN]
                pivot_pct.plot(kind="bar", stacked=True, ax=ax,
                               color=colors_stack[:len(pivot_pct.columns)],
                               edgecolor=DARK_BG, linewidth=0.3)
                ax.set_xlabel("Shipping Mode", color=TEXT_CLR, fontsize=9)
                ax.set_ylabel("Percentage (%)", color=TEXT_CLR, fontsize=9)
                ax.set_title("Delivery Status by Shipping Mode (%)", color=TEXT_CLR, fontsize=11, fontweight="600")
                ax.legend(fontsize=7, labelcolor=TEXT_CLR, facecolor=DARK_AX,
                          edgecolor="#334155", bbox_to_anchor=(1.02, 1))
                plt.xticks(rotation=30, ha="right")
                style_fig(fig)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
                st.caption("💡 **First Class** is nearly 100% late. **Standard Class** has a healthier mix.")
            except Exception:
                pass

    st.markdown("---")

    # Correlation heatmap
    st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    num_df = raw_df.select_dtypes(include=np.number)
    drop_corr = [c for c in ["Latitude","Longitude"] if c in num_df.columns]
    num_df = num_df.drop(columns=drop_corr, errors="ignore")
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                    ax=ax, linewidths=0.5, linecolor="#334155",
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
        ax.set_title("Correlation Matrix of Numerical Features", color=TEXT_CLR, fontsize=12, fontweight="600")
        style_fig(fig)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        if "Late_delivery_risk" in corr.columns:
            top_corr = corr["Late_delivery_risk"].drop("Late_delivery_risk").abs().sort_values(ascending=False).head(5)
            st.caption(f"💡 Top correlates with Late Delivery Risk: **{', '.join(top_corr.index)}**")

    st.markdown("---")

    # Shipping days analysis
    if all(c in raw_df.columns for c in ["Days for shipping (real)","Days for shipment (scheduled)","Shipping Mode"]):
        st.markdown('<div class="section-title">⏱ Shipping Days Analysis</div>', unsafe_allow_html=True)
        ship_agg = raw_df.groupby("Shipping Mode").agg(
            actual=("Days for shipping (real)", "mean"),
            scheduled=("Days for shipment (scheduled)", "mean")
        ).reset_index()
        ship_agg["variance"] = ship_agg["actual"] - ship_agg["scheduled"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(ship_agg["Shipping Mode"], ship_agg["actual"],    "o-", color=ACC_BLUE, label="Actual",    lw=2)
        axes[0].plot(ship_agg["Shipping Mode"], ship_agg["scheduled"], "o-", color=ACC_AMB,  label="Scheduled", lw=2)
        axes[0].set_title("Actual vs Scheduled Shipping Days", color=TEXT_CLR, fontsize=10, fontweight="600")
        axes[0].set_xlabel("Shipping Mode", color=TEXT_CLR); axes[0].set_ylabel("Avg Days", color=TEXT_CLR)
        axes[0].legend(fontsize=8, labelcolor=TEXT_CLR, facecolor=DARK_AX, edgecolor="#334155")
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=25, ha="right")

        axes[1].plot(ship_agg["Shipping Mode"], ship_agg["variance"], "o-", color=ACC_RED, lw=2)
        axes[1].fill_between(ship_agg["Shipping Mode"], ship_agg["variance"],
                             alpha=0.25, color=ACC_RED)
        axes[1].set_title("Delivery Date Variance by Shipping Mode", color=TEXT_CLR, fontsize=10, fontweight="600")
        axes[1].set_xlabel("Shipping Mode", color=TEXT_CLR); axes[1].set_ylabel("Variance (days)", color=TEXT_CLR)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=25, ha="right")
        style_fig(fig, axes)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ────────────────────────────────────────────────────────────
#  TAB 3 — ML MODELS
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">🤖 Machine Learning Model Training & Evaluation</div>',
                unsafe_allow_html=True)

    with st.spinner("⚙️ Training Random Forest, XGBoost, Logistic Regression..."):
        try:
            results = train_models(raw_df)
            training_ok = True
        except Exception as e:
            st.error(f"Model training failed: {e}")
            training_ok = False

    if training_ok:
        y_test    = results["y_test"]
        y_pred_rf = results["y_pred_rf"]
        y_pred_xgb= results["y_pred_xgb"]
        y_pred_lr = results["y_pred_lr"]

        xgb_label = "XGBoost" if XGBOOST_OK else "Gradient Boosting (XGBoost fallback)"

        # ── Random Forest ──────────────────────────────────
        st.markdown("### 🌳 Random Forest Classifier")
        st.markdown('<div class="model-box">', unsafe_allow_html=True)

        rf_acc = accuracy_score(y_test, y_pred_rf)
        rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
        rf_rec  = recall_score(y_test, y_pred_rf, zero_division=0)
        rf_f1   = f1_score(y_test, y_pred_rf, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{rf_acc:.4f}")
        c2.metric("Precision", f"{rf_prec:.4f}")
        c3.metric("Recall",    f"{rf_rec:.4f}")
        c4.metric("F1-Score",  f"{rf_f1:.4f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Classification Report")
            report_rf = classification_report(y_test, y_pred_rf,
                                              target_names=["On-Time (0)","Late (1)"],
                                              output_dict=True)
            report_df = pd.DataFrame(report_rf).T.round(4)
            st.dataframe(report_df, use_container_width=True)
            st.caption("""
**Interpretation:**  
- Precision for Late class → of all predicted late orders, how many were truly late  
- Recall for Late class → of all actual late orders, how many were caught  
- High recall for class 1 is critical to avoid missing real late deliveries
            """)

        with col_b:
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig = confusion_matrix_fig(cm_rf, "Random Forest — Confusion Matrix")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.subheader("Feature Importance — Top 15")
        fig = feature_importance_fig(results["fi_rf"], "Random Forest Feature Importance", color=ACC_BLUE)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption(
            "💡 `Days for shipping (real)` and `Order_to_Shipment_Time` dominate — "
            "logistics timing is the #1 driver of late delivery risk."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── XGBoost ────────────────────────────────────────
        st.markdown(f"### ⚡ {xgb_label}")
        st.markdown('<div class="model-box">', unsafe_allow_html=True)

        xgb_acc  = accuracy_score(y_test, y_pred_xgb)
        xgb_prec = precision_score(y_test, y_pred_xgb, zero_division=0)
        xgb_rec  = recall_score(y_test, y_pred_xgb, zero_division=0)
        xgb_f1   = f1_score(y_test, y_pred_xgb, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{xgb_acc:.4f}")
        c2.metric("Precision", f"{xgb_prec:.4f}")
        c3.metric("Recall",    f"{xgb_rec:.4f}")
        c4.metric("F1-Score",  f"{xgb_f1:.4f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Classification Report")
            report_xgb = classification_report(y_test, y_pred_xgb,
                                               target_names=["On-Time (0)","Late (1)"],
                                               output_dict=True)
            report_df_xgb = pd.DataFrame(report_xgb).T.round(4)
            st.dataframe(report_df_xgb, use_container_width=True)
            st.caption("""
**Interpretation:**  
- XGBoost often matches or exceeds Random Forest on structured data  
- `Days for shipment (scheduled)` is the top XGBoost predictor — planned timelines matter
            """)

        with col_b:
            cm_xgb = confusion_matrix(y_test, y_pred_xgb)
            fig = confusion_matrix_fig(cm_xgb, f"{xgb_label} — Confusion Matrix")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.subheader("Feature Importance — Top 15")
        fig = feature_importance_fig(results["fi_xgb"], f"{xgb_label} Feature Importance", color="#a78bfa")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption(
            "💡 `Days for shipment (scheduled)` appears as top feature in XGBoost — "
            "longer planned windows reduce perceived late risk."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Model Comparison ───────────────────────────────
        st.markdown("### 📊 Model Comparison")
        lr_acc  = accuracy_score(y_test, y_pred_lr)
        lr_prec = precision_score(y_test, y_pred_lr, zero_division=0)
        lr_rec  = recall_score(y_test, y_pred_lr, zero_division=0)
        lr_f1   = f1_score(y_test, y_pred_lr, zero_division=0)

        names = ["Logistic Reg.", "Random Forest", xgb_label]
        accs  = [lr_acc,  rf_acc,  xgb_acc]
        precs = [lr_prec, rf_prec, xgb_prec]
        recs  = [lr_rec,  rf_rec,  xgb_rec]
        f1s   = [lr_f1,   rf_f1,   xgb_f1]

        fig = model_comparison_fig(names, accs, precs, recs, f1s)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        comp_df = pd.DataFrame({
            "Model": names,
            "Accuracy":  [f"{v:.4f}" for v in accs],
            "Precision": [f"{v:.4f}" for v in precs],
            "Recall":    [f"{v:.4f}" for v in recs],
            "F1-Score":  [f"{v:.4f}" for v in f1s],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        best = names[int(np.argmax(f1s))]
        st.success(f"🏆 Best model by F1-Score: **{best}**")


# ────────────────────────────────────────────────────────────
#  TAB 4 — SUPPLIERS
# ────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🏭 Supplier Reliability Dashboard</div>',
                unsafe_allow_html=True)

    # Derive supplier proxy from available columns
    supplier_col = None
    for c in ["Supplier", "Department Name", "Category Name", "Market"]:
        if c in raw_df.columns:
            supplier_col = c
            break

    if supplier_col is None:
        st.info("No supplier-level column detected. Showing Shipping Mode analysis.")
        supplier_col = "Shipping Mode" if "Shipping Mode" in raw_df.columns else None

    if supplier_col and "Late_delivery_risk" in raw_df.columns:
        agg = raw_df.groupby(supplier_col).agg(
            total_orders=("Late_delivery_risk", "count"),
            late_orders=("Late_delivery_risk", "sum"),
        ).reset_index()
        agg["on_time_orders"]  = agg["total_orders"] - agg["late_orders"]
        agg["late_rate_%"]     = (agg["late_orders"] / agg["total_orders"] * 100).round(1)
        agg["on_time_rate_%"]  = (100 - agg["late_rate_%"]).round(1)
        if "Sales" in raw_df.columns:
            sales_by = raw_df.groupby(supplier_col)["Sales"].sum().reset_index()
            agg = agg.merge(sales_by, on=supplier_col, how="left")
            agg["Sales"] = agg["Sales"].fillna(0).round(0).astype(int)

        agg = agg.sort_values("late_rate_%", ascending=True).reset_index(drop=True)

        def reliability_label(r):
            if r < 30:   return "🟢 Excellent"
            elif r < 55: return "🟡 Moderate"
            else:        return "🔴 High Risk"

        agg["Reliability"] = agg["late_rate_%"].apply(reliability_label)

        # Summary cards
        c1, c2, c3 = st.columns(3)
        excellent = (agg["late_rate_%"] < 30).sum()
        moderate  = ((agg["late_rate_%"] >= 30) & (agg["late_rate_%"] < 55)).sum()
        high_risk = (agg["late_rate_%"] >= 55).sum()
        c1.metric("🟢 Excellent Suppliers", excellent)
        c2.metric("🟡 Moderate Suppliers",  moderate)
        c3.metric("🔴 High-Risk Suppliers", high_risk)

        st.markdown("---")

        # Table
        st.markdown(f"### 📋 {supplier_col} Reliability Ranking")
        display_cols = [supplier_col, "total_orders", "on_time_orders", "late_orders",
                        "on_time_rate_%", "late_rate_%", "Reliability"]
        if "Sales" in agg.columns:
            display_cols.insert(-1, "Sales")

        st.dataframe(
            agg[display_cols].rename(columns={
                supplier_col: supplier_col,
                "total_orders":  "Total Orders",
                "on_time_orders":"On-Time",
                "late_orders":   "Late",
                "on_time_rate_%":"On-Time %",
                "late_rate_%":   "Late %",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Chart: On-time rate by supplier
        col_a, col_b = st.columns(2)
        with col_a:
            top_n = agg.sort_values("on_time_rate_%", ascending=False).head(10)
            colors = [ACC_GRN if v >= 60 else ACC_AMB if v >= 45 else ACC_RED
                      for v in top_n["on_time_rate_%"]]
            fig = bar_chart(top_n[supplier_col], top_n["on_time_rate_%"],
                            f"Top 10 by On-Time Rate ({supplier_col})",
                            supplier_col, "On-Time %", color=colors, horiz=True)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            worst = agg.sort_values("late_rate_%", ascending=False).head(10)
            colors2 = [ACC_RED if v >= 55 else ACC_AMB for v in worst["late_rate_%"]]
            fig = bar_chart(worst[supplier_col], worst["late_rate_%"],
                            f"Top 10 Highest Late Rate ({supplier_col})",
                            supplier_col, "Late %", color=colors2, horiz=True)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # If Days available
        if "Days for shipping (real)" in raw_df.columns:
            st.markdown("---")
            st.markdown(f"### ⏱ Average Actual Shipping Days by {supplier_col}")
            days_agg = raw_df.groupby(supplier_col)["Days for shipping (real)"].mean().sort_values()
            fig = bar_chart(days_agg.index, days_agg.values,
                            f"Avg Actual Shipping Days by {supplier_col}",
                            supplier_col, "Avg Days",
                            color=[ACC_GRN if v < 4 else ACC_AMB if v < 6 else ACC_RED
                                   for v in days_agg.values])
            st.pyplot(fig, use_container_width=True)
            plt.close()

    else:
        st.warning("Upload a dataset with a 'Late_delivery_risk' column to see supplier analysis.")


# ────────────────────────────────────────────────────────────
#  TAB 5 — PREDICT
# ────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">🔮 Predict Late Delivery Risk</div>',
                unsafe_allow_html=True)

    if not training_ok:
        st.error("Models failed to train. Please check your dataset.")
    else:
        sub1, sub2 = st.tabs(["Single Shipment", "Batch Prediction"])

        # ── Single Prediction ──────────────────────────────
        with sub1:
            st.markdown("#### Enter Shipment Details")
            c1, c2, c3 = st.columns(3)

            with c1:
                days_real  = st.number_input("Days for Shipping (Actual)", 1, 15, 4)
                days_sched = st.number_input("Days for Shipping (Scheduled)", 1, 10, 4)
                ship_mode  = st.selectbox("Shipping Mode",
                    ["Standard Class","Second Class","First Class","Same Day"])

            with c2:
                payment    = st.selectbox("Payment Type", ["DEBIT","TRANSFER","CASH","PAYMENT"])
                order_qty  = st.number_input("Order Item Quantity", 1, 20, 1)
                prod_price = st.number_input("Product Price ($)", 5.0, 2000.0, 99.99)

            with c3:
                sales      = st.number_input("Sales ($)", 1.0, 5000.0, 200.0)
                benefit    = st.number_input("Benefit per Order ($)", -500.0, 1000.0, 50.0)
                profit     = st.number_input("Order Profit per Order ($)", -300.0, 500.0, 30.0)

            col_extra1, col_extra2 = st.columns(2)
            with col_extra1:
                order_to_ship = st.number_input(
                    "Order-to-Shipment Time (hrs)", 0, 200, 24,
                    help="Hours from order placed to shipment"
                )
            with col_extra2:
                ship_dow = st.selectbox("Day of Week Shipped",
                    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

            dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
            ship_mode_enc = {"Standard Class":0,"Second Class":1,"First Class":2,"Same Day":3}

            if st.button("🚀 Predict Now", type="primary", use_container_width=True):
                # Build a minimal feature vector matching training features
                rf_model = results["rf"]
                feat_names = results["feature_names"]
                scaler     = results["scaler"]
                num_cols_m = results["num_cols"]

                # Create zero vector
                input_dict = {f: 0.0 for f in feat_names}

                # Fill known features
                mapping = {
                    "Days for shipping (real)":       days_real,
                    "Days for shipment (scheduled)":  days_sched,
                    "Shipping Mode":                  ship_mode_enc.get(ship_mode, 0),
                    "Order Item Quantity":             order_qty,
                    "Product Price":                  prod_price,
                    "Sales":                          sales,
                    "Benefit per order":              benefit,
                    "Order Profit Per Order":         profit,
                    "Order_to_Shipment_Time":         order_to_ship,
                    "ship_day_of_week":               dow_map.get(ship_dow, 0),
                    "Order Item Total":               prod_price * order_qty,
                    "Sales per customer":             sales,
                    "Order Item Product Price":       prod_price,
                }
                for k, v in mapping.items():
                    if k in input_dict:
                        input_dict[k] = v

                # Payment type OHE columns
                for pt in ["DEBIT","TRANSFER","CASH","PAYMENT"]:
                    col_name = f"Type_{pt}"
                    if col_name in input_dict:
                        input_dict[col_name] = 1.0 if payment == pt else 0.0

                input_df = pd.DataFrame([input_dict])
                # Scale numeric
                for nc in num_cols_m:
                    if nc in input_df.columns:
                        pass  # will scale below
                try:
                    sc_cols = [c for c in num_cols_m if c in input_df.columns]
                    if sc_cols:
                        input_df[sc_cols] = scaler.transform(input_df[sc_cols])
                except Exception:
                    pass

                pred_rf  = rf_model.predict(input_df)[0]
                prob_rf  = rf_model.predict_proba(input_df)[0][1]
                pred_xgb = results["xgb"].predict(input_df)[0]
                prob_xgb = results["xgb"].predict_proba(input_df)[0][1]

                is_late = (pred_rf == 1 or pred_xgb == 1)
                avg_prob = (prob_rf + prob_xgb) / 2

                st.markdown("---")
                st.markdown("#### 🎯 Prediction Result")

                if is_late:
                    st.markdown(f"""
                    <div class="pred-result-late">
                      <h2>⚠️ HIGH LATE DELIVERY RISK</h2>
                      <p>This shipment is <strong>likely to be delayed</strong>. Average risk probability: <strong>{avg_prob:.1%}</strong></p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="pred-result-ok">
                      <h2>✅ LOW LATE DELIVERY RISK</h2>
                      <p>This shipment is <strong>expected to arrive on time</strong>. Risk probability: <strong>{avg_prob:.1%}</strong></p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("---")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("🌳 Random Forest Prediction",
                              "⚠️ LATE" if pred_rf == 1 else "✅ ON-TIME",
                              f"Confidence: {prob_rf:.1%}")
                with col_r2:
                    xgb_lbl = "XGBoost" if XGBOOST_OK else "Gradient Boosting"
                    st.metric(f"⚡ {xgb_lbl} Prediction",
                              "⚠️ LATE" if pred_xgb == 1 else "✅ ON-TIME",
                              f"Confidence: {prob_xgb:.1%}")

                if days_real > days_sched:
                    st.warning(f"⚠️ Actual shipping days ({days_real}) exceed scheduled ({days_sched}) — this is a strong indicator of late delivery.")
                if ship_mode == "First Class":
                    st.warning("⚠️ First Class shipping has the highest late delivery rate in this dataset.")
                if order_to_ship > 48:
                    st.warning(f"⚠️ Order-to-shipment time of {order_to_ship}hrs is high — consider streamlining fulfillment.")

        # ── Batch Prediction ───────────────────────────────
        with sub2:
            st.markdown("#### Upload a file for batch prediction")
            st.info("Upload a CSV or Excel with the same columns as your training data (without the 'Late_delivery_risk' column). Predictions will be appended.")

            batch_file = st.file_uploader(
                "Upload batch data (CSV or Excel only)",
                type=["csv", "xlsx"],
                key="batch_upload"
            )

            if batch_file:
                if not batch_file.name.lower().endswith((".csv", ".xlsx")):
                    st.error("❌ Only .csv or .xlsx files are accepted.")
                else:
                    try:
                        fb = batch_file.read()
                        if batch_file.name.endswith(".csv"):
                            batch_df = pd.read_csv(io.BytesIO(fb), encoding="latin1")
                        else:
                            batch_df = pd.read_excel(io.BytesIO(fb))

                        st.success(f"Loaded {len(batch_df):,} rows.")

                        # Add dummy target for preprocessing
                        had_target = "Late_delivery_risk" in batch_df.columns
                        if not had_target:
                            batch_df["Late_delivery_risk"] = 0

                        proc = preprocess(batch_df)
                        feat_names = results["feature_names"]
                        # Align columns
                        for f in feat_names:
                            if f not in proc.columns:
                                proc[f] = 0.0
                        proc = proc[feat_names + (["Late_delivery_risk"] if "Late_delivery_risk" in proc.columns else [])]

                        X_b = proc[[c for c in feat_names if c in proc.columns]]
                        sc_cols_b = [c for c in results["num_cols"] if c in X_b.columns]
                        try:
                            if sc_cols_b:
                                X_b[sc_cols_b] = results["scaler"].transform(X_b[sc_cols_b])
                        except Exception:
                            pass

                        batch_df["RF_Prediction"]  = results["rf"].predict(X_b)
                        batch_df["XGB_Prediction"] = results["xgb"].predict(X_b)
                        batch_df["RF_Late_Prob"]   = results["rf"].predict_proba(X_b)[:, 1].round(3)
                        batch_df["XGB_Late_Prob"]  = results["xgb"].predict_proba(X_b)[:, 1].round(3)
                        batch_df["Risk_Label"] = np.where(
                            (batch_df["RF_Prediction"] == 1) | (batch_df["XGB_Prediction"] == 1),
                            "⚠️ LATE RISK", "✅ ON-TIME"
                        )

                        st.markdown("#### Prediction Results")
                        st.dataframe(batch_df.head(50), use_container_width=True)

                        late_count = ((batch_df["RF_Prediction"] == 1) | (batch_df["XGB_Prediction"] == 1)).sum()
                        st.metric("Orders at Late Risk", f"{late_count:,} / {len(batch_df):,}",
                                  f"{late_count/len(batch_df)*100:.1f}%")

                        csv_out = batch_df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Predictions CSV",
                            data=csv_out,
                            file_name="predictions_output.csv",
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"Error processing batch file: {e}")

# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.78rem;'>"
    "Supply Chain Intelligence Hub &nbsp;·&nbsp; "
    "DataCo Smart Supply Chain Dataset &nbsp;·&nbsp; "
    "Models: Logistic Regression · Random Forest · XGBoost"
    "</p>",
    unsafe_allow_html=True,
)
