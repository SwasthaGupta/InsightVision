from pathlib import Path
import pandas as pd
import streamlit as st

from modules.data_analysis import analyze_dataframe, build_eda_figures
from modules.ml_model import infer_task, train_models
from modules.vision import VisionEngine
from modules.multimodal import match_images_to_data, summarize_clusters
from modules.ai_insights import generate_insights, payload_signature
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# PREMIUM PLOTLY CHART THEME
# =========================================================
def style_plotly_chart(fig, title=None, height=390):
    """Apply a consistent, lightweight purple analytics style."""
    if fig is None:
        return fig

    fig.update_layout(
        title=dict(
            text=title if title else None,
            font=dict(
                family="Palatino Linotype, Book Antiqua, Georgia, serif",
                size=18,
                color="#28143f"
            ),
            x=0,
            xanchor="left",
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.76)",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#35204f"
        ),
        margin=dict(l=55, r=35, t=58, b=48),
        hoverlabel=dict(
            bgcolor="#24103f",
            bordercolor="#6d3bb3",
            font=dict(color="#ffffff", size=12)
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(103,62,150,.12)",
            borderwidth=1,
            font=dict(color="#35204f")
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="rgba(80,45,120,.16)",
        tickfont=dict(color="#5e5270"),
        title_font=dict(color="#35204f"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(110,70,160,.10)",
        zeroline=False,
        showline=False,
        tickfont=dict(color="#5e5270"),
        title_font=dict(color="#35204f"),
    )

    for trace in fig.data:
        if trace.type == "bar":
            trace.update(
                width=0.52,
                marker_line_width=0,
                opacity=0.88,
            )
        elif trace.type == "scatter":
            trace.update(
                marker=dict(size=7),
                line=dict(width=1.5),
            )

    return fig


def show_chart(fig, title=None, height=390):
    fig = style_plotly_chart(fig, title=title, height=height)
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )


st.set_page_config(page_title="InsightVision", page_icon=None, layout="wide", initial_sidebar_state="expanded")

# here 1
st.markdown("""
<style>

    /* ==========================================
       UPLOADED FILE NAME
       ========================================== */

    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] * {
        color: #17121f !important;
        -webkit-text-fill-color: #17121f !important;
        opacity: 1 !important;
    }

    /* File size */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"] * {
        color: #4a4352 !important;
        -webkit-text-fill-color: #4a4352 !important;
        opacity: 1 !important;
    }

    /* Uploaded file card */
    [data-testid="stFileUploaderFile"] {
        background: #f7f7f9 !important;
        border-radius: 12px !important;
    }

    /* All text inside uploaded file card */
    [data-testid="stFileUploaderFile"] p,
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] div {
        color: #17121f !important;
        -webkit-text-fill-color: #17121f !important;
    }

</style>
""", unsafe_allow_html=True)

# ---------- Visual theme ----------
st.markdown("""
<style>
:root {
    --purple-950: #170b2e;
    --purple-900: #24103f;
    --purple-800: #35145f;
    --purple-700: #4b1f78;
    --purple-600: #6d3bb3;
    --purple-500: #8f5de7;
    --purple-300: #c9b3ff;
    --ink: #221735;
    --muted: #6f6480;
    --card: rgba(255,255,255,.96);
}

/* Main app background */
.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(177, 129, 255, .28), transparent 34%),
        radial-gradient(circle at 90% 15%, rgba(117, 67, 190, .25), transparent 30%),
        linear-gradient(135deg, #f7f2ff 0%, #efe7fb 45%, #f8f5ff 100%);
    color: var(--ink);
}

/* Hide default Streamlit chrome */
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #211035 0%, #30134f 48%, #4a2075 100%);
    border-right: 1px solid rgba(255,255,255,.12);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption { color: #f7f1ff !important; }
section[data-testid="stSidebar"] .stRadio label { color: #f7f1ff !important; opacity: .95; }
section[data-testid="stSidebar"] .stRadio label:hover { color: #ffffff !important; }

/* Sidebar uploaders */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #3b1761 0%, #5a2387 48%, #7b4cc2 100%) !important;
    border: 1px dashed rgba(255,255,255,.62) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 24px rgba(20, 8, 40, .20) !important;
}

/* Only the actual upload/dropzone controls use white text. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Uploaded-file card: white background + dark readable text. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {
    background-color: #ffffff !important;
    color: #17121f !important;
    -webkit-text-fill-color: #17121f !important;
    opacity: 1 !important;
    text-shadow: none !important;
}

/* Do not paint the card's icon background white. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] img {
    background: transparent !important;
}

/* Filename */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] [class*="file-name"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] [class*="FileName"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] [title] {
    color: #17121f !important;
    -webkit-text-fill-color: #17121f !important;
    opacity: 1 !important;
    font-weight: 500 !important;
    text-shadow: none !important;
}

/* File size */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileData"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileData"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] [class*="file-size"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] [class*="FileSize"] {
    color: #4a4352 !important;
    -webkit-text-fill-color: #4a4352 !important;
    opacity: 1 !important;
}

/* Catch native Streamlit versions that use class names rather than test IDs. */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [class*="uploadedFile"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [class*="UploadedFile"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [class*="file-name"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [class*="file-size"] {
    color: #17121f !important;
    -webkit-text-fill-color: #17121f !important;
    opacity: 1 !important;
}

/* Keep the remove button visible and purple. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button * {
    background: transparent !important;
    color: #5d2a8a !important;
    -webkit-text-fill-color: #5d2a8a !important;
    opacity: 1 !important;
}


    /* Extra selector coverage for Streamlit's uploaded-file row. */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] [data-testid="stMarkdownContainer"] *,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] [role="button"] span {
        color: #17121f !important;
        -webkit-text-fill-color: #17121f !important;
        opacity: 1 !important;
    }

/* Upload button */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #6d3bb3 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 0 !important;
}

.upload-preview {
    display:flex; align-items:center; gap:.65rem; margin:.35rem 0 .75rem;
    padding:.55rem .65rem; border-radius:12px;
    background:rgba(255,255,255,.96); border:1px solid rgba(185,154,221,.45);
    box-shadow:0 5px 14px rgba(18,7,35,.12);
}
.upload-preview-icon {
    width:30px; height:30px; border-radius:8px; display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#6d3bb3,#9b6be4); color:white; font-weight:800;
}
.upload-preview strong { display:block; color:#2b1742 !important; font-size:.78rem; line-height:1.1; }
.upload-preview small { display:block; color:#756985 !important; font-size:.7rem; margin-top:.12rem; }
.image-preview-title { color:#eadfff; font-size:.72rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; margin:.25rem 0 .45rem; }
.image-name { color:#f4ecff !important; font-size:.62rem; line-height:1.1; text-align:center; margin-top:.18rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
section[data-testid="stSidebar"] [data-testid="stImage"] img { border-radius:10px !important; border:1px solid rgba(255,255,255,.28) !important; box-shadow:0 4px 12px rgba(10,4,20,.18); }

/* Main content width / headings */
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3 { color: #27133e !important; letter-spacing: -.02em; }

/* Hero */
.hero {
    padding: 1.45rem 1.6rem;
    border-radius: 24px;
    color: white;
    background: linear-gradient(115deg, #24103f 0%, #4d2379 48%, #7650b5 100%);
    box-shadow: 0 18px 50px rgba(67, 32, 111, .25);
    margin-bottom: 1.25rem;
}
.hero h1 { color: white !important; margin: 0; font-size: 2.35rem; }
.hero p { color: #e9ddff; margin: .35rem 0 0; font-size: 1rem; }
.hero-badge {
    display: inline-block;
    margin-bottom: .55rem;
    padding: .35rem .7rem;
    border-radius: 999px;
    background: rgba(255,255,255,.13);
    border: 1px solid rgba(255,255,255,.2);
    color: #f4edff;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
}

/* Cards */
.metric-card, .insight-card, .info-card {
    background: var(--card);
    border: 1px solid rgba(103, 62, 150, .14);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 30px rgba(70, 37, 105, .09);
}
.metric-label { color: var(--muted); font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
.metric-value { color: #35145f; font-size: 1.8rem; font-weight: 800; margin-top: .2rem; }
.metric-help { color: #81758f; font-size: .78rem; margin-top: .1rem; }

/* Streamlit metric fallback styling */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.92);
    border: 1px solid rgba(103,62,150,.14);
    padding: 1rem;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(70,37,105,.08);
}
div[data-testid="stMetricLabel"] { color: #6f6480 !important; }
div[data-testid="stMetricValue"] { color: #35145f !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #5d2a8a, #8b58d8);
    color: white;
    border: 0;
    border-radius: 12px;
    font-weight: 700;
    padding: .55rem 1.1rem;
    box-shadow: 0 8px 20px rgba(93,42,138,.25);
    transition: all .15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 26px rgba(93,42,138,.32); }

/* Inputs */
div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    border-radius: 12px !important;
    border-color: #d9cceb !important;
}

/* Tabs / expanders */
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] { color: #6d5b7e; font-weight: 700; }
.stTabs [aria-selected="true"] { color: #5b2789 !important; }
.streamlit-expanderHeader { border-radius: 14px; }

/* Tables */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(103,62,150,.13);
    box-shadow: 0 8px 24px rgba(70,37,105,.07);
}

/* Alerts */
div[data-testid="stAlert"] { border-radius: 14px; }

.section-kicker {
    color: #7141a7;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: .2rem;
}
.section-title { color: #28143f; font-size: 1.45rem; font-weight: 800; margin-bottom: .8rem; }
.small-note { color: #756985; font-size: .82rem; }

.pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: .55rem;
    margin: .8rem 0 1.2rem;
}
.pipeline span {
    background: rgba(109,59,179,.1);
    color: #4d2379;
    border: 1px solid rgba(109,59,179,.16);
    border-radius: 999px;
    padding: .38rem .7rem;
    font-size: .78rem;
    font-weight: 700;
}

.insight {
    background: linear-gradient(135deg, rgba(255,255,255,.97), rgba(247,241,255,.97));
    border-left: 4px solid #7a4bc1;
    border-radius: 0 14px 14px 0;
    padding: .8rem 1rem;
    margin: .55rem 0;
    box-shadow: 0 8px 22px rgba(70,37,105,.06);
}

/* Elegant typography for page headings */
h1, h2, h3, .section-title, .hero h1 {
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif !important;
    font-weight: 800 !important;
}
.stMarkdown h3 { font-size: 1.08rem !important; }
.section-title { letter-spacing: -.015em; }

/* Compact, refined chart surfaces */
div[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,.34);
    border: 1px solid rgba(103,62,150,.09);
    border-radius: 14px;
    padding: 0.15rem 0.15rem 0.05rem;
    box-shadow: 0 7px 20px rgba(70,37,105,.055);
}
</style>
""", unsafe_allow_html=True)


def hero(title="InsightVision", subtitle="Multimodal Data & Visual Analytics Platform", kicker="AI-powered analytics"):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-badge">{kicker}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-help">{help_text}</div>
            </div>
            """, unsafe_allow_html=True)


hero()

with st.sidebar:
    st.markdown("### Data workspace")
    data_file = st.file_uploader("CSV or Excel dataset", type=["csv", "xlsx", "xls"])
    image_files = st.file_uploader("Images (optional)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    st.markdown("### Explore")
    page = st.radio("Navigation", ["Overview", "Data Analysis", "ML Analysis", "Vision Analysis", "Multimodal Analysis", "AI Insights"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("InsightVision • Multimodal analytics workspace")

if not data_file:
    st.info("Upload a CSV/XLSX dataset to begin.")
    st.markdown("<div class='pipeline'><span>Data</span><span>EDA</span><span>ML</span><span>ResNet50</span><span>Clustering</span><span>Multimodal</span><span>AI Insights</span></div>", unsafe_allow_html=True)
    st.stop()

try:
    df = pd.read_csv(data_file) if data_file.name.lower().endswith(".csv") else pd.read_excel(data_file)
except Exception as e:
    st.error(f"Could not read dataset: {e}")
    st.stop()

analysis = analyze_dataframe(df)

if page == "Overview":
    st.markdown("<div class='section-kicker'>Command center</div><div class='section-title'>Dataset overview</div>", unsafe_allow_html=True)
    metric_cards([
        ("Rows", f"{len(df):,}", "Records available for analysis"),
        ("Columns", f"{len(df.columns):,}", "Features and business fields"),
        ("Missing cells", f"{analysis['missing_cells']:,}", "Values requiring attention"),
        ("Duplicates", f"{analysis['duplicates']:,}", "Duplicate records detected"),
    ])
    st.markdown("### Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("### Automatic summary")
    for x in analysis["highlights"]:
        st.markdown(f"<div class='insight'>{x}</div>", unsafe_allow_html=True)

elif page == "Data Analysis":
    st.markdown("<div class='section-kicker'>Explore patterns</div><div class='section-title'>Data Analysis</div>", unsafe_allow_html=True)
    metric_cards([
        ("Numeric fields", len(analysis["numeric_cols"]), "Continuous or count-based variables"),
        ("Categorical fields", len(analysis["categorical_cols"]), "Labels and business dimensions"),
        ("Missing cells", analysis["missing_cells"], "Data quality signal"),
    ])
    st.markdown("### Missing-value profile")
    if analysis["missing_df"].empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(analysis["missing_df"], use_container_width=True)
    for title, fig in build_eda_figures(df):
        st.markdown(f"### {title}")
        show_chart(fig, title=title, height=380)

elif page == "ML Analysis":
    st.markdown("<div class='section-kicker'>Predictive analytics</div><div class='section-title'>ML Analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("Target column", df.columns, index=list(df.columns).index("Profit") if "Profit" in df.columns else 0)
    with c2:
        task = st.selectbox("Task", ["Auto", "Regression", "Classification"], index=1 if "Profit" in df.columns else 0)
    st.caption("Tip: choose a business outcome such as Profit for regression rather than an identifier such as Product_ID.")
    if st.button("Train models", type="primary"):
        try:
            with st.spinner("Training models..."):
                result = train_models(df, target, infer_task(df[target], task))
            st.session_state["ml_result"] = result
        except Exception as e:
            st.error(f"Training failed: {e}")
    result = st.session_state.get("ml_result")
    if result:
        score_col = "R2" if result["task"] == "Regression" else "Accuracy"
        best_row = result["metrics"].sort_values(score_col, ascending=False).iloc[0]
        metric_cards([
            ("Task", result["task"], "Selected modeling problem"),
            ("Best model", best_row["Model"], f"Highest {score_col}"),
            (score_col, f"{best_row[score_col]:.4f}", "Test-set performance"),
        ])
        st.markdown("### Model comparison")
        st.dataframe(result["metrics"], use_container_width=True)
        show_chart(result["comparison_fig"], title="Model Performance — Comparison", height=370)
        st.markdown("### Feature importance")
        show_chart(result["importance_fig"], title="Feature Importance", height=400)
        st.success(result["best_model_text"])

elif page == "Vision Analysis":
    st.markdown("<div class='section-kicker'>Computer vision</div><div class='section-title'>Vision Analysis</div>", unsafe_allow_html=True)
    if not image_files:
        st.warning("Upload images first."); st.stop()
    if st.button("Process images", type="primary"):
        temp = Path("data/processed/uploaded_images"); temp.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in image_files:
            p = temp / f.name; p.write_bytes(f.getbuffer()); paths.append(str(p))
        try:
            with st.spinner("Extracting ResNet50 embeddings..."):
                eng = VisionEngine(); emb, meta = eng.extract_embeddings(paths)
            st.session_state["vision_embeddings"] = emb
            st.session_state["vision_metadata"] = meta
            st.session_state["vision_engine"] = eng
        except Exception as e:
            st.error(f"Vision processing failed: {e}")
    emb = st.session_state.get("vision_embeddings")
    meta = st.session_state.get("vision_metadata")
    if emb is not None:
        k = st.slider("Visual clusters", 2, min(10, len(meta)), min(4, len(meta)))
        labels, fig = st.session_state["vision_engine"].cluster(emb, k)
        st.session_state["vision_labels"] = labels
        metric_cards([
            ("Images", len(meta), "Images successfully processed"),
            ("Embedding size", emb.shape[1], "ResNet50 feature dimensions"),
            ("Visual clusters", k, "K-Means groups"),
        ])
        show_chart(fig, title="Image Embedding Clusters", height=390)
        st.markdown("### Visual gallery")
        cols = st.columns(4)
        for i, item in enumerate(meta[:20]):
            with cols[i % 4]:
                st.image(item["path"], caption=f"{Path(item['path']).name}  •  Cluster {labels[i]}", use_container_width=True)
        st.markdown("### Similar-image search")
        selected = st.selectbox("Find similar to", range(len(meta)), format_func=lambda i: Path(meta[i]["path"]).name)
        sim_cols = st.columns(3)
        for col, (idx, score) in zip(sim_cols, st.session_state["vision_engine"].similar(emb, selected, 3)):
            with col:
                st.image(meta[idx]["path"], caption=f"Similarity: {score:.3f}", use_container_width=True)

elif page == "Multimodal Analysis":
    st.markdown("<div class='section-kicker'>Data + vision fusion</div><div class='section-title'>Multimodal Analysis</div>", unsafe_allow_html=True)
    emb = st.session_state.get("vision_embeddings")
    meta = st.session_state.get("vision_metadata")
    labels = st.session_state.get("vision_labels")
    if emb is None or labels is None:
        st.warning("Process images in Vision Analysis first."); st.stop()
    matches = match_images_to_data(df, meta)
    if matches.empty:
        st.warning("No matches found. Example: Product_ID=101 and image filename 101.jpg."); st.stop()
    matches["cluster"] = labels[matches["image_index"].to_numpy()]
    nums = df.select_dtypes("number").columns.tolist()
    summary = summarize_clusters(matches, nums)
    st.markdown("### Cluster-level business performance")
    st.dataframe(summary, use_container_width=True)
    if "Profit" in summary.columns:
        best_profit = summary.loc[summary["Profit"].idxmax()]
        st.info(f"Cluster {int(best_profit['cluster'])} has the highest average profit: {best_profit['Profit']:,.0f}.")
    if "Sales" in summary.columns:
        best_sales = summary.loc[summary["Sales"].idxmax()]
        st.info(f"Cluster {int(best_sales['cluster'])} has the highest average sales: {best_sales['Sales']:,.0f}.")
    st.markdown("### Matched records")
    st.dataframe(matches.head(50), use_container_width=True)

elif page == "AI Insights":
    st.markdown(
        "<div class='section-kicker'>Explainable intelligence</div>"
        "<div class='section-title'>AI Insights</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Business-oriented conclusions are generated from the computed dataset, "
        "ML, vision and multimodal results. The AI layer does not invent or calculate "
        "the underlying metrics."
    )

    # Pass authoritative dataset facts directly so the AI layer never falls back
    # to missing/default values such as 0 rows or 0 columns.
    useful_numeric = df.select_dtypes(include="number").columns.tolist()
    useful_numeric_non_id = [
        c for c in useful_numeric
        if not ("id" in str(c).lower() and df[c].nunique(dropna=True) / max(len(df), 1) > 0.5)
    ]
    highest_variance_metric = None
    if useful_numeric_non_id:
        variance = df[useful_numeric_non_id].var().sort_values(ascending=False)
        if len(variance):
            highest_variance_metric = variance.index[0]

    payload = {
        "dataset": {
            **analysis,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "numeric_count": int(len(df.select_dtypes(include="number").columns)),
            "categorical_count": int(len(df.select_dtypes(exclude="number").columns)),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "highest_variance_metric": highest_variance_metric,
        }
    }

    if "ml_result" in st.session_state:
        r = st.session_state["ml_result"]
        metrics_records = r["metrics"].to_dict("records")
        best_row = r["metrics"].sort_values(
            "R2" if r["task"] == "Regression" else "Accuracy",
            ascending=False
        ).iloc[0]
        payload["ml"] = {
            "task": r["task"],
            "best_model": best_row["Model"],
            "best_score": best_row["R2"] if r["task"] == "Regression" else best_row["Accuracy"],
            "metrics": metrics_records,
        }

    if "vision_labels" in st.session_state and "vision_metadata" in st.session_state:
        payload["vision"] = {
            "images": len(st.session_state["vision_metadata"]),
            "clusters": len(set(st.session_state["vision_labels"])),
        }

    # Re-create the multimodal summary from the current session so AI Insights
    # can reason over the actual image-cluster/business-metric relationships.
    emb = st.session_state.get("vision_embeddings")
    meta = st.session_state.get("vision_metadata")
    labels = st.session_state.get("vision_labels")
    if emb is not None and labels is not None and meta is not None:
        try:
            matches_ai = match_images_to_data(df, meta)
            if not matches_ai.empty:
                matches_ai["cluster"] = labels[matches_ai["image_index"].to_numpy()]
                nums_ai = df.select_dtypes("number").columns.tolist()
                summary_ai = summarize_clusters(matches_ai, nums_ai)
                payload["multimodal"] = {
                    "matched_records": len(matches_ai),
                    "summary": summary_ai,
                }
        except Exception:
            pass

    # Invalidate old AI text whenever the authoritative dataset/ML/vision/multimodal
    # results change. This prevents stale values such as an old cluster count or R²
    # from surviving in Streamlit session state.
    current_signature = payload_signature(payload)
    if st.session_state.get("insights_payload_signature") != current_signature:
        st.session_state.pop("insights", None)

    if st.button("Generate insights", type="primary"):
        with st.spinner("Generating business insights..."):
            st.session_state["insights"] = generate_insights(payload)
            st.session_state["insights_payload_signature"] = current_signature

    insights = st.session_state.get("insights")

    if insights:
        st.markdown("### Executive summary")
        st.markdown(
            f"<div class='insight-card' style='border-left:5px solid #7a4bc1;'>"
            f"<div style='font-size:1rem; line-height:1.7; color:#2b1742;'>"
            f"{insights.get('summary', '')}"
            f"</div></div>",
            unsafe_allow_html=True
        )

        findings = insights.get("findings", [])
        recommendations = insights.get("recommendations", [])
        warnings = insights.get("warnings", [])

        if findings:
            st.markdown("### Key findings")
            for item in findings:
                st.markdown(
                    f"<div class='insight'>• {item}</div>",
                    unsafe_allow_html=True
                )

        if recommendations:
            st.markdown("### Business recommendations")
            for item in recommendations:
                st.markdown(
                    f"<div class='insight' style='border-left-color:#5d2a8a;'>"
                    f"• {item}</div>",
                    unsafe_allow_html=True
                )

        if warnings:
            st.markdown("### Data & model considerations")
            for item in warnings:
                st.warning(item)

        source_label = (
            "LLM generated from computed results"
            if insights.get("source") == "llm_from_computed_results"
            else "Local rule-based insights from computed results"
        )
        st.caption(source_label)
