"""
House Price Prediction — Linear Regression
==========================================
Run:  streamlit run app.py
Deps: pip install streamlit altair scikit-learn pandas numpy
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURES = ["SquareFootage", "Bedrooms", "Bathrooms", "Age", "GarageSpaces", "DistanceToCity"]

FEATURE_LABELS = {
    "SquareFootage":  "Square Footage (sq ft)",
    "Bedrooms":       "Bedrooms",
    "Bathrooms":      "Bathrooms",
    "Age":            "Property Age (years)",
    "GarageSpaces":   "Garage Spaces",
    "DistanceToCity": "Distance to City (km)",
}

FEATURE_ICONS = {
    "SquareFootage":  "📐",
    "Bedrooms":       "🛏️",
    "Bathrooms":      "🚿",
    "Age":            "🏚️",
    "GarageSpaces":   "🚗",
    "DistanceToCity": "📍",
}

SLIDER_CFG = {
    # (min, max, default, step)
    "SquareFootage":  (600,  4000, 1800, 50),
    "Bedrooms":       (1,    5,    3,    1),
    "Bathrooms":      (1,    4,    2,    1),
    "Age":            (1,    60,   10,   1),
    "GarageSpaces":   (0,    3,    1,    1),
    "DistanceToCity": (1.0,  50.0, 12.5, 0.5),
}

SCENARIOS = {
    "🏠 Starter": [950,  2, 1, 18, 1, 22.0],
    "🏡 Family":  [1800, 3, 2, 10, 1, 12.5],
    "🏰 Luxury":  [3600, 5, 4,  4, 3,  6.0],
}

# ── Styles ────────────────────────────────────────────────────────────────────

def inject_styles() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    #MainMenu, footer, header  { visibility: hidden; }

    .stApp { background: #0d1117; color: #e6edf3; }

    section[data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    .hero { padding: 1.8rem 0 1.2rem; border-bottom: 1px solid #21262d; margin-bottom: 1.5rem; }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5f3fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15; margin: 0;
    }
    .hero-sub { color: #8b949e; font-size: 0.95rem; margin-top: 0.3rem; font-weight: 300; }

    .kpi-row   { display: flex; gap: 0.9rem; margin-bottom: 1.3rem; flex-wrap: wrap; }
    .kpi-card  {
        flex: 1; min-width: 110px;
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 0.9rem 1.1rem;
    }
    .kpi-card .lbl { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .06em; }
    .kpi-card .val { color: #e6edf3; font-size: 1.45rem; font-weight: 600; margin-top: .1rem; }
    .kpi-card .sub { color: #3fb950; font-size: 0.78rem; margin-top: .05rem; }

    .price-banner {
        background: linear-gradient(135deg, #0d1b2a, #112240);
        border: 1px solid #1f6feb; border-radius: 14px;
        padding: 1.6rem 2rem; text-align: center; margin-bottom: 1.1rem;
    }
    .price-banner .plbl { color: #79c0ff; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .1em; }
    .price-banner .pval {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem; color: #ffffff; line-height: 1.1; margin: .2rem 0;
    }
    .price-banner .prng { color: #8b949e; font-size: 0.85rem; }
    .dpos { color: #3fb950; } .dneg { color: #f85149; }

    .icard {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: .8rem 1rem; margin-bottom: .6rem;
    }
    .icard .il { color: #8b949e; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; }
    .icard .iv { color: #e6edf3; font-size: 1rem; font-weight: 500; }

    .stTabs [data-baseweb="tab-list"] {
        background: #161b22; border-radius: 8px;
        border: 1px solid #30363d; padding: .2rem; gap: .15rem;
    }
    .stTabs [data-baseweb="tab"]      { border-radius: 6px; color: #8b949e !important; padding: .4rem .9rem; font-size: .87rem; }
    .stTabs [aria-selected="true"]    { background: #21262d !important; color: #e6edf3 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: #fff !important; border: none !important; border-radius: 8px !important;
        font-weight: 600 !important; padding: .55rem 1.2rem !important; font-size: .93rem !important;
    }
    .stButton > button:hover { opacity: .88 !important; }

    p, span, div, li, td, th, label { color: #c9d1d9 !important; }
    h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        font-weight: 400 !important; color: #e6edf3 !important;
    }
    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.2rem  !important; }
    hr { border-color: #21262d !important; margin: 1.1rem 0 !important; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.35rem !important; }
    [data-testid="stMetricLabel"] { color: #8b949e  !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic dataset matching original project (500 records, same formula)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "SquareFootage":  rng.integers(600, 4000, n),
        "Bedrooms":       rng.integers(1, 6, n),
        "Bathrooms":      rng.integers(1, 5, n),
        "Age":            rng.integers(1, 60, n),
        "GarageSpaces":   rng.integers(0, 4, n),
        "DistanceToCity": rng.uniform(1, 50, n).round(1),
    })
    df["Price"] = (
          145  * df["SquareFootage"]
        + 9_000  * df["Bedrooms"]
        + 12_500 * df["Bathrooms"]
        - 550    * df["Age"]
        + 7_000  * df["GarageSpaces"]
        - 1_400  * df["DistanceToCity"]
        + rng.normal(0, 18_000, n)
    ).clip(50_000).round(2)
    return df


# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def train_model(split: int = 80):
    df = build_dataset()
    X, y = df[FEATURES], df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - split / 100, random_state=42
    )
    y_test = y_test.reset_index(drop=True)

    # StandardScaler — same as original project
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        "r2":      r2_score(y_test, y_pred),
        "mae":     mean_absolute_error(y_test, y_pred),
        "rmse":    float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "res_std": float(np.std(y_test.values - y_pred)),
    }
    coef_df = pd.DataFrame({
        "Feature":     FEATURES,
        "Coefficient": model.coef_,
    })
    return model, scaler, df, X_test, y_test, y_pred, metrics, coef_df


# ── Session state ─────────────────────────────────────────────────────────────

def init_state() -> None:
    for key, (_, _, default, _) in SLIDER_CFG.items():
        st.session_state.setdefault(key, default)
    st.session_state.setdefault("pred_result", None)


def load_scenario(name: str) -> None:
    for key, val in zip(FEATURES, SCENARIOS[name]):
        st.session_state[key] = val
    st.session_state.pred_result = None


# ── Prediction ────────────────────────────────────────────────────────────────

def run_prediction(model, scaler, df, metrics) -> None:
    vals   = [st.session_state[f] for f in FEATURES]
    X_in   = scaler.transform(np.array(vals).reshape(1, -1))
    price  = max(0.0, float(model.predict(X_in)[0]))
    avg    = float(df["Price"].mean())
    ivl    = 1.96 * metrics["res_std"]

    # Feature contribution vs dataset mean (unscaled, interpretable)
    means  = df[FEATURES].mean()
    delta  = pd.Series(dict(zip(FEATURES, vals))) - means
    # Use original (unscaled) coef approximation via mean price decomposition
    # We re-fit a raw LR for interpretability display only
    raw_lr = LinearRegression().fit(df[FEATURES], df["Price"])
    impact = delta * raw_lr.coef_

    contrib_df = pd.DataFrame({
        "Feature":     FEATURES,
        "PriceImpact": impact.values,
        "Input":       vals,
        "Mean":        means.values,
    })

    st.session_state.pred_result = {
        "price":      price,
        "avg":        avg,
        "lower":      max(0.0, price - ivl),
        "upper":      price + ivl,
        "delta_pct":  (price - avg) / avg * 100,
        "contrib_df": contrib_df,
    }


# ── Tab 1 — Predictor ─────────────────────────────────────────────────────────

def tab_predictor(model, scaler, df, metrics) -> None:
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.markdown("### Property Details")
        for feat in FEATURES:
            mn, mx, _, step = SLIDER_CFG[feat]
            st.slider(
                f"{FEATURE_ICONS[feat]}  {FEATURE_LABELS[feat]}",
                mn, mx, key=feat, step=step,
            )
        st.markdown("")
        if st.button("🏷️  Predict Price", use_container_width=True):
            run_prediction(model, scaler, df, metrics)

    with right:
        pr = st.session_state.pred_result

        if pr:
            dcls  = "dpos" if pr["delta_pct"] >= 0 else "dneg"
            dsign = "+" if pr["delta_pct"] >= 0 else ""
            st.markdown(f"""
            <div class="price-banner">
                <div class="plbl">Estimated Market Value</div>
                <div class="pval">${pr['price']:,.0f}</div>
                <div class="prng">95 % range &nbsp;·&nbsp; ${pr['lower']:,.0f} — ${pr['upper']:,.0f}</div>
                <div class="{dcls}" style="margin-top:.4rem;font-size:.9rem;">
                    {dsign}{pr['delta_pct']:.1f}% vs dataset average
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            diff = pr["price"] - pr["avg"]
            with c1:
                st.markdown(f'<div class="icard"><div class="il">Dataset Average</div>'
                            f'<div class="iv">${pr["avg"]:,.0f}</div></div>', unsafe_allow_html=True)
            with c2:
                s = "+" if diff >= 0 else ""
                st.markdown(f'<div class="icard"><div class="il">Difference</div>'
                            f'<div class="iv">{s}${diff:,.0f}</div></div>', unsafe_allow_html=True)

            st.markdown("**Feature impact vs average**")
            cdf = pr["contrib_df"]
            bar = (
                alt.Chart(cdf)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                          cornerRadiusTopLeft=4, cornerRadiusBottomLeft=4)
                .encode(
                    x=alt.X("PriceImpact:Q", title="Price impact ($)",
                             axis=alt.Axis(format="$,.0f", labelColor="#8b949e",
                                           titleColor="#8b949e", gridColor="#21262d")),
                    y=alt.Y("Feature:N", sort="-x",
                             axis=alt.Axis(labelColor="#c9d1d9", titleColor="#8b949e")),
                    color=alt.condition(
                        alt.datum.PriceImpact > 0,
                        alt.value("#3fb950"), alt.value("#f85149"),
                    ),
                    tooltip=[
                        "Feature",
                        alt.Tooltip("PriceImpact:Q", title="Impact ($)", format="$,.0f"),
                        alt.Tooltip("Input:Q",       title="Your value",  format=",.1f"),
                        alt.Tooltip("Mean:Q",        title="Dataset mean",format=",.1f"),
                    ],
                )
                .properties(height=230, background="transparent")
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(bar, use_container_width=True)

        else:
            st.markdown("""
            <div style="background:#161b22;border:1px dashed #30363d;border-radius:12px;
                        padding:2.5rem;text-align:center;color:#8b949e;">
                <div style="font-size:2.5rem">🏠</div>
                <div style="margin-top:.7rem;font-size:.95rem;">
                    Adjust the sliders and click
                    <strong style="color:#58a6ff;">Predict Price</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Input vs mean table — always visible
        st.markdown("**Your inputs vs dataset averages**")
        means = df[FEATURES].mean()
        comp  = pd.DataFrame({
            "Feature":      [FEATURE_LABELS[f] for f in FEATURES],
            "Your Input":   [st.session_state[f] for f in FEATURES],
            "Dataset Mean": [round(means[f], 1) for f in FEATURES],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True, height=250)


# ── Tab 2 — Model Results ─────────────────────────────────────────────────────

def tab_results(y_test, y_pred, metrics, coef_df) -> None:
    st.markdown("### Model Evaluation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score",     f"{metrics['r2']:.4f}")
    c2.metric("MAE",          f"${metrics['mae']:,.0f}")
    c3.metric("RMSE",         f"${metrics['rmse']:,.0f}")
    c4.metric("Residual Std", f"${metrics['res_std']:,.0f}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("**Actual vs Predicted Prices**")
        mn = float(min(y_test.min(), y_pred.min()))
        mx = float(max(y_test.max(), y_pred.max()))
        sc_df  = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        ref_df = pd.DataFrame({"x": [mn, mx], "y": [mn, mx]})

        scatter = (
            alt.Chart(sc_df)
            .mark_circle(size=38, opacity=0.5, color="#388bfd")
            .encode(
                x=alt.X("Actual:Q",    title="Actual Price ($)",
                         axis=alt.Axis(format="$~s", labelColor="#8b949e",
                                       titleColor="#8b949e", gridColor="#21262d")),
                y=alt.Y("Predicted:Q", title="Predicted Price ($)",
                         axis=alt.Axis(format="$~s", labelColor="#8b949e",
                                       titleColor="#8b949e", gridColor="#21262d")),
                tooltip=[alt.Tooltip("Actual:Q",    format="$,.0f"),
                          alt.Tooltip("Predicted:Q", format="$,.0f")],
            )
        )
        ref = (
            alt.Chart(ref_df)
            .mark_line(color="#f0e040", strokeDash=[6, 4], strokeWidth=1.8)
            .encode(x="x:Q", y="y:Q")
        )
        st.altair_chart(
            (scatter + ref)
            .properties(height=290, background="transparent")
            .configure_view(strokeOpacity=0),
            use_container_width=True,
        )

    with right:
        st.markdown("**Residual Distribution**")
        resid_df = pd.DataFrame({"Residual": y_test.values - y_pred})
        hist = (
            alt.Chart(resid_df)
            .mark_bar(color="#3fb950", opacity=0.78, binSpacing=1)
            .encode(
                x=alt.X("Residual:Q", bin=alt.Bin(maxbins=35), title="Residual ($)",
                         axis=alt.Axis(format="$~s", labelColor="#8b949e",
                                       titleColor="#8b949e", gridColor="#21262d")),
                y=alt.Y("count()", title="Count",
                         axis=alt.Axis(labelColor="#8b949e", titleColor="#8b949e",
                                       gridColor="#21262d")),
                tooltip=[alt.Tooltip("Residual:Q", bin=True, format="$,.0f"),
                          alt.Tooltip("count()", title="Count")],
            )
            .properties(height=290, background="transparent")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(hist, use_container_width=True)

    st.markdown("---")
    st.markdown("**Feature Coefficients — Linear Regression**")
    coef_chart = (
        alt.Chart(coef_df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                  cornerRadiusTopLeft=4, cornerRadiusBottomLeft=4)
        .encode(
            x=alt.X("Coefficient:Q", title="Coefficient value",
                     axis=alt.Axis(labelColor="#8b949e", titleColor="#8b949e",
                                   gridColor="#21262d")),
            y=alt.Y("Feature:N", sort="-x",
                     axis=alt.Axis(labelColor="#c9d1d9", titleColor="#8b949e")),
            color=alt.condition(
                alt.datum.Coefficient > 0,
                alt.value("#3fb950"), alt.value("#f85149"),
            ),
            tooltip=["Feature", alt.Tooltip("Coefficient:Q", format=",.2f")],
        )
        .properties(height=250, background="transparent")
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(coef_chart, use_container_width=True)


# ── Tab 3 — EDA ───────────────────────────────────────────────────────────────

def tab_eda(df: pd.DataFrame) -> None:
    st.markdown("### Exploratory Data Analysis")

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown("**Dataset Preview** (first 100 rows)")
        st.dataframe(df.head(100), use_container_width=True, height=270)
        st.markdown("**Descriptive Statistics**")
        stats = df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
        st.dataframe(stats, use_container_width=True)

    with right:
        feat = st.selectbox("Select feature to plot against Price", FEATURES, index=0)
        sample = df.sample(min(500, len(df)), random_state=7)

        scatter = (
            alt.Chart(sample)
            .mark_circle(size=42, opacity=0.4, color="#58a6ff")
            .encode(
                x=alt.X(f"{feat}:Q", title=FEATURE_LABELS[feat],
                         axis=alt.Axis(labelColor="#8b949e", titleColor="#8b949e",
                                       gridColor="#21262d")),
                y=alt.Y("Price:Q", title="Price ($)",
                         axis=alt.Axis(format="$~s", labelColor="#8b949e",
                                       titleColor="#8b949e", gridColor="#21262d")),
                tooltip=[alt.Tooltip(feat, format=",.1f"),
                          alt.Tooltip("Price:Q", format="$,.0f")],
            )
        )
        trend = (
            scatter.transform_regression(feat, "Price")
            .mark_line(color="#f0e040", strokeWidth=2.5)
        )
        st.altair_chart(
            (scatter + trend)
            .properties(height=300, background="transparent")
            .configure_view(strokeOpacity=0),
            use_container_width=True,
        )

        st.markdown("**Correlation Matrix**")
        corr      = df.corr(numeric_only=True).round(3)
        corr_long = corr.reset_index().melt(id_vars="index")
        corr_long.columns = ["Feature A", "Feature B", "Correlation"]
        heat = (
            alt.Chart(corr_long)
            .mark_rect()
            .encode(
                x=alt.X("Feature A:N",
                         axis=alt.Axis(labelColor="#c9d1d9", labelAngle=-30, titleOpacity=0)),
                y=alt.Y("Feature B:N",
                         axis=alt.Axis(labelColor="#c9d1d9", titleOpacity=0)),
                color=alt.Color("Correlation:Q",
                    scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
                    legend=alt.Legend(labelColor="#8b949e", titleColor="#8b949e")),
                tooltip=["Feature A", "Feature B",
                          alt.Tooltip("Correlation:Q", format=".3f")],
            )
            .properties(height=250, background="transparent")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(heat, use_container_width=True)

    st.markdown("**Price Distribution**")
    ph = (
        alt.Chart(df)
        .mark_bar(color="#388bfd", opacity=0.8, binSpacing=2)
        .encode(
            x=alt.X("Price:Q", bin=alt.Bin(maxbins=40), title="Price ($)",
                     axis=alt.Axis(format="$~s", labelColor="#8b949e",
                                   titleColor="#8b949e", gridColor="#21262d")),
            y=alt.Y("count()", title="Count",
                     axis=alt.Axis(labelColor="#8b949e", titleColor="#8b949e",
                                   gridColor="#21262d")),
            tooltip=[alt.Tooltip("Price:Q", bin=True, format="$,.0f"),
                      alt.Tooltip("count()", title="Count")],
        )
        .properties(height=210, background="transparent")
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(ph, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="House Price Prediction — Linear Regression",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    init_state()

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">House Price Prediction</div>
        <div class="hero-sub">Linear Regression · StandardScaler · R² 0.979 · 500 records · 6 features</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        split = st.slider("Train / Test split (%)", 70, 90, 80, 5)
        st.markdown("---")
        st.markdown("## 🏘️ Preset Scenarios")
        for name in SCENARIOS:
            if st.button(name, use_container_width=True):
                load_scenario(name)
                st.rerun()
        st.markdown("---")
        st.markdown("""
        <div style="color:#8b949e;font-size:.78rem;line-height:1.7;">
        <b style="color:#c9d1d9;">Algorithm</b><br>Linear Regression<br><br>
        <b style="color:#c9d1d9;">Preprocessing</b><br>StandardScaler<br><br>
        <b style="color:#c9d1d9;">Dataset</b><br>500 synthetic records<br><br>
        <b style="color:#c9d1d9;">Author</b><br>Venkata Sriharika Prathipati<br>
        </div>
        """, unsafe_allow_html=True)

    # Train
    with st.spinner("Training model…"):
        model, scaler, df, X_test, y_test, y_pred, metrics, coef_df = train_model(split)

    # KPI strip
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="lbl">R² Score</div>
        <div class="val">{metrics['r2']:.4f}</div>
        <div class="sub">↑ excellent fit</div></div>
      <div class="kpi-card"><div class="lbl">MAE</div>
        <div class="val">${metrics['mae']:,.0f}</div></div>
      <div class="kpi-card"><div class="lbl">RMSE</div>
        <div class="val">${metrics['rmse']:,.0f}</div></div>
      <div class="kpi-card"><div class="lbl">Records</div>
        <div class="val">500</div>
        <div class="sub">6 features</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    t1, t2, t3 = st.tabs(["🏷️  Price Predictor", "📊  Model Results", "🔍  Data Explorer"])
    with t1:
        tab_predictor(model, scaler, df, metrics)
    with t2:
        tab_results(y_test, y_pred, metrics, coef_df)
    with t3:
        tab_eda(df)


if __name__ == "__main__":
    main()
