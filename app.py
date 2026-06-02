import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "SquareFootage",
    "Bedrooms",
    "Bathrooms",
    "Age",
    "GarageSpaces",
    "DistanceToCity",
]

SCENARIOS = {
    "Starter Home": [950, 2, 1, 18, 1, 22.0],
    "Family Home": [1800, 3, 2, 10, 1, 12.5],
    "Luxury Home": [3600, 5, 4, 4, 3, 6.0],
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>

        /* Global App */
        .stApp {
            background-color: #f8fafc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Typography */
        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .main-header {
            color: #0f172a;
            font-size: 2.8rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.3rem;
        }

        .sub-header {
            text-align: center;
            color: #64748b;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: white;
            border-right: 1px solid #e2e8f0;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #0f172a;
        }

        /* Cards */
        .card {
            background: white;
            border-radius: 14px;
            padding: 1.5rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(15,23,42,0.05);
            margin-bottom: 1rem;
        }

        .card * {
            color: #0f172a !important;
        }

        /* Metrics */
        [data-testid="metric-container"] {
            background: white;
            border: 1px solid #e2e8f0;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(15,23,42,0.05);
        }

        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 700;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            border: none;
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            height: 3rem;
            transition: 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #1d4ed8;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            border-radius: 8px;
        }

        .stTabs [aria-selected="true"] {
            color: #2563eb !important;
        }

        /* Dataframes */
        [data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }

        /* Slider Labels */
        label {
            font-weight: 600 !important;
            color: #334155 !important;
        }

        .hint {
            color: #64748b;
            font-size: 0.9rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def build_dataset(seed: int = 42, n_samples: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "SquareFootage": rng.integers(600, 4000, n_samples),
            "Bedrooms": rng.integers(1, 6, n_samples),
            "Bathrooms": rng.integers(1, 5, n_samples),
            "Age": rng.integers(1, 60, n_samples),
            "GarageSpaces": rng.integers(0, 4, n_samples),
            "DistanceToCity": rng.uniform(1, 50, n_samples),
        }
    )
    data["Price"] = (
        145 * data["SquareFootage"]
        + 9000 * data["Bedrooms"]
        + 12500 * data["Bathrooms"]
        - 550 * data["Age"]
        + 7000 * data["GarageSpaces"]
        - 1400 * data["DistanceToCity"]
        + rng.normal(0, 18000, n_samples)
    ).clip(50000)
    return data


@st.cache_resource(show_spinner=False)
def train_model(train_ratio: int = 80):
    df = build_dataset()
    x = df[FEATURES]
    y = df["Price"]
    test_size = 1.0 - (train_ratio / 100)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "residual_std": float(np.std(y_test - y_pred)),
    }
    coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": model.coef_})
    return model, df, x_test, y_test, y_pred, metrics, coef_df, float(model.intercept_)


def init_state() -> None:
    if "scenario" not in st.session_state:
        st.session_state.scenario = "Family Home"
    defaults = SCENARIOS[st.session_state.scenario]
    for key, value in zip(FEATURES, defaults):
        st.session_state.setdefault(key, value)


def load_scenario(name: str) -> None:
    st.session_state.scenario = name
    values = SCENARIOS[name]
    for key, value in zip(FEATURES, values):
        st.session_state[key] = value


def predict(model: LinearRegression) -> float:
    sample = pd.DataFrame(
        [[st.session_state[f] for f in FEATURES]],
        columns=FEATURES,
    )
    return float(model.predict(sample)[0])


def build_contributions(
    current_values: pd.Series,
    means: pd.Series,
    coef_df: pd.DataFrame,
    intercept: float,
    avg_price: float,
) -> pd.DataFrame:
    joined = coef_df.set_index("Feature").join(current_values.rename("Input")).join(means.rename("Mean"))
    joined["DeltaFromMean"] = joined["Input"] - joined["Mean"]
    joined["PriceImpact"] = joined["DeltaFromMean"] * joined["Coefficient"]
    result = joined.reset_index()[["Feature", "Input", "Mean", "Coefficient", "PriceImpact"]]
    # Calibration row to map linear decomposition close to predicted value.
    calibration = avg_price - (intercept + (means * joined["Coefficient"]).sum())
    result.loc[len(result)] = ["Calibration", np.nan, np.nan, np.nan, calibration]
    return result


def render_predictor_tab(
    model: LinearRegression,
    df: pd.DataFrame,
    avg_price: float,
    metrics: dict,
    coef_df: pd.DataFrame,
    intercept: float,
) -> None:
    st.subheader("Real-Time House Price Predictor")
    left, right = st.columns(2)
    with left:
        st.write("Adjust the property details:")
        st.slider("Square Footage", 600, 4000, key="SquareFootage", step=50)
        st.slider("Bedrooms", 1, 5, key="Bedrooms")
        st.slider("Bathrooms", 1, 4, key="Bathrooms")
        st.slider("Age (Years)", 1, 60, key="Age")
        st.slider("Garage Spaces", 0, 3, key="GarageSpaces")
        st.slider("Distance to City (km)", 1.0, 50.0, key="DistanceToCity")
        do_predict = st.button("Predict Price", use_container_width=True, type="primary")

    with right:
        if do_predict:
            pred_price = max(0.0, predict(model))
            delta_pct = ((pred_price - avg_price) / avg_price) * 100
            interval = 1.96 * metrics["residual_std"]
            lower = max(0.0, pred_price - interval)
            upper = pred_price + interval
            st.metric("AI Estimated Price", f"${pred_price:,.2f}", f"{delta_pct:+.1f}% vs Avg")
            st.caption(f"95% expected range: ${lower:,.0f} to ${upper:,.0f}")
        else:
            st.info("Click **Predict Price** to generate valuation.")

        st.write("Input vs Dataset Means")
        means = df[FEATURES].mean().rename("Mean")
        current = pd.Series({f: st.session_state[f] for f in FEATURES}, name="Your Input")
        comp = pd.concat([current, means], axis=1).reset_index().rename(columns={"index": "Feature"})
        st.dataframe(comp, use_container_width=True, hide_index=True)

        if do_predict:
            st.write("Why this price? (Feature contribution)")
            contrib = build_contributions(current, means, coef_df, intercept, avg_price)
            contrib_chart = (
                alt.Chart(contrib[contrib["Feature"] != "Calibration"])
                .mark_bar()
                .encode(
                    x=alt.X("PriceImpact:Q", title="Impact on Price vs Average"),
                    y=alt.Y("Feature:N", sort="-x"),
                    color=alt.condition(
                        alt.datum.PriceImpact > 0,
                        alt.value("#16a34a"),
                        alt.value("#dc2626"),
                    ),
                    tooltip=[
                        "Feature",
                        alt.Tooltip("PriceImpact:Q", format="$,.0f"),
                        alt.Tooltip("Input:Q", format=",.2f"),
                        alt.Tooltip("Mean:Q", format=",.2f"),
                    ],
                )
                .properties(height=210)
            )
            st.altair_chart(contrib_chart, use_container_width=True)
      


def render_analytics_tab(
    coef_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    metrics: dict,
) -> None:
    st.subheader("Model Performance & Diagnostics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{metrics['r2']:.4f}")
    m2.metric("MAE", f"${metrics['mae']:,.0f}")
    m3.metric("RMSE", f"${metrics['rmse']:,.0f}")
    m4.metric("Residual Std", f"${metrics['residual_std']:,.0f}")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Feature Coefficients")
        coef_chart = (
            alt.Chart(coef_df)
            .mark_bar()
            .encode(
                x=alt.X("Coefficient:Q", title="Coefficient Impact"),
                y=alt.Y("Feature:N", sort="-x"),
                color=alt.condition(
                    alt.datum.Coefficient > 0,
                    alt.value("#10b981"),
                    alt.value("#ef4444"),
                ),
                tooltip=["Feature", alt.Tooltip("Coefficient:Q", format=",.2f")],
            )
            .properties(height=260)
        )
        st.altair_chart(coef_chart, use_container_width=True)

    with c2:
        st.write("Actual vs Predicted")
        plot_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        scatter = (
            alt.Chart(plot_df)
            .mark_circle(size=42, opacity=0.55, color="#2563eb")
            .encode(
                x=alt.X("Actual:Q", title="Actual Price"),
                y=alt.Y("Predicted:Q", title="Predicted Price"),
                tooltip=[alt.Tooltip("Actual:Q", format="$,.0f"), alt.Tooltip("Predicted:Q", format="$,.0f")],
            )
            .properties(height=260)
        )
        st.altair_chart(scatter, use_container_width=True)


def render_explorer_tab(df: pd.DataFrame) -> None:
    st.subheader("Interactive Dataset Explorer")
    l, r = st.columns([1, 1.1])
    with l:
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(80), use_container_width=True, height=300)
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe().T[["mean", "std", "min", "max"]], use_container_width=True)
    with r:
        feature = st.selectbox("X-axis Feature", FEATURES, index=0)
        sample = df.sample(min(500, len(df)), random_state=42)
        scatter = (
            alt.Chart(sample)
            .mark_circle(size=45, opacity=0.45, color="#2563eb")
            .encode(
                x=alt.X(f"{feature}:Q", title=feature),
                y=alt.Y("Price:Q", title="Price"),
                tooltip=[feature, alt.Tooltip("Price:Q", format="$,.0f")],
            )
        )
        trend = scatter.transform_regression(feature, "Price").mark_line(color="#2563eb", strokeWidth=3)
        st.altair_chart((scatter + trend).properties(height=360), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Valora | AI House Price Predictor",
        layout="wide",
        page_icon="🏠",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    init_state()

    st.markdown('<div class="main-header">ValuEdge | AI House Price Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Premium-style ML valuation app with live prediction, diagnostics, and data explorer.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Preset Scenarios")
        train_ratio = st.slider("Training Split (%)", min_value=70, max_value=90, value=80, step=5)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Starter", use_container_width=True):
                load_scenario("Starter Home")
                st.rerun()
            if st.button("Luxury", use_container_width=True):
                load_scenario("Luxury Home")
                st.rerun()
        with c2:
            if st.button("Family", use_container_width=True):
                load_scenario("Family Home")
                st.rerun()
            if st.button("Reset", use_container_width=True):
                load_scenario("Family Home")
                st.rerun()
        st.markdown('<p class="hint">Use presets to create quick demo interactions.</p>', unsafe_allow_html=True)
        st.caption("Accessibility: all controls have labels and keyboard focus support.")

    model, df, x_test, y_test, y_pred, metrics, coef_df, intercept = train_model(train_ratio=train_ratio)
    avg_price = float(df["Price"].mean())
    t1, t2, t3 = st.tabs(["Price Predictor", "Model Diagnostics", "Data Explorer"])

    with t1:
        render_predictor_tab(model, df, avg_price, metrics, coef_df, intercept)
    with t2:
        render_analytics_tab(coef_df, y_test, y_pred, metrics)
    with t3:
        render_explorer_tab(df)


if __name__ == "__main__":
    main()
