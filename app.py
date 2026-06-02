from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "SquareFootage",
    "Bedrooms",
    "Bathrooms",
    "Age",
    "GarageSpaces",
    "DistanceToCity",
]

SCENARIOS = {
    "Starter Home": {
        "SquareFootage": 950,
        "Bedrooms": 2,
        "Bathrooms": 1,
        "Age": 18,
        "GarageSpaces": 1,
        "DistanceToCity": 22.0,
    },
    "Family Home": {
        "SquareFootage": 1800,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Age": 10,
        "GarageSpaces": 1,
        "DistanceToCity": 12.5,
    },
    "Luxury Home": {
        "SquareFootage": 3600,
        "Bedrooms": 5,
        "Bathrooms": 4,
        "Age": 4,
        "GarageSpaces": 3,
        "DistanceToCity": 6.0,
    },
}


@dataclass
class TrainedBundle:
    model: LinearRegression
    scaler: StandardScaler
    r2: float
    mae: float
    baseline_price: float
    coef_df: pd.DataFrame
    feature_stats: pd.DataFrame


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #eef3fb 100%); }
            .hero {
                border-radius: 18px;
                padding: 1rem 1.2rem;
                background: linear-gradient(120deg, #132a53, #2563eb);
                color: white;
                box-shadow: 0 10px 24px rgba(19, 42, 83, 0.25);
                margin-bottom: 1rem;
            }
            .hero h1 { margin: 0; font-size: 1.7rem; }
            .hero p { margin: 0.3rem 0 0; opacity: 0.96; }
            .glass {
                background: rgba(255, 255, 255, 0.85);
                border: 1px solid rgba(37, 99, 235, 0.1);
                border-radius: 14px;
                padding: 0.9rem;
                transition: transform 180ms ease, box-shadow 180ms ease;
            }
            .glass:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(15, 23, 42, 0.12); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_dataset(seed: int = 42, n_samples: int = 500) -> pd.DataFrame:
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
    noise = rng.normal(0, 20000, n_samples)
    data["Price"] = (
        150 * data["SquareFootage"]
        + 8000 * data["Bedrooms"]
        + 12000 * data["Bathrooms"]
        - 500 * data["Age"]
        + 7000 * data["GarageSpaces"]
        - 1500 * data["DistanceToCity"]
        + noise
    ).clip(50000)
    return data


@st.cache_resource(show_spinner=False)
def train_pipeline() -> TrainedBundle:
    data = build_dataset()
    x = data[FEATURE_COLUMNS].copy()
    y = data["Price"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    coef_df = (
        pd.DataFrame({"Feature": FEATURE_COLUMNS, "Coefficient": model.coef_})
        .sort_values("Coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )
    feature_stats = x.describe().loc[["mean", "std"]].T.rename(
        columns={"mean": "Mean", "std": "StdDev"}
    )

    return TrainedBundle(
        model=model,
        scaler=scaler,
        r2=model.score(x_test_scaled, y_test),
        mae=mean_absolute_error(y_test, y_pred),
        baseline_price=float(y.mean()),
        coef_df=coef_df,
        feature_stats=feature_stats,
    )


def init_state() -> None:
    default = SCENARIOS["Family Home"]
    for key, value in default.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_scenario(scenario_name: str) -> None:
    scenario = SCENARIOS[scenario_name]
    for key, value in scenario.items():
        st.session_state[key] = value


def collect_inputs() -> Dict[str, float]:
    return {
        "SquareFootage": float(st.session_state["SquareFootage"]),
        "Bedrooms": float(st.session_state["Bedrooms"]),
        "Bathrooms": float(st.session_state["Bathrooms"]),
        "Age": float(st.session_state["Age"]),
        "GarageSpaces": float(st.session_state["GarageSpaces"]),
        "DistanceToCity": float(st.session_state["DistanceToCity"]),
    }


def predict_price(bundle: TrainedBundle, input_row: Dict[str, float]) -> Tuple[float, float]:
    frame = pd.DataFrame([input_row])[FEATURE_COLUMNS]
    scaled = bundle.scaler.transform(frame)
    predicted = float(bundle.model.predict(scaled)[0])
    # Lightweight uncertainty proxy using MAE for confidence band.
    confidence_band = 1.15 * bundle.mae
    return predicted, confidence_band


def render_app() -> None:
    st.set_page_config(page_title="ValuEdge AI Prediction", page_icon="🏠", layout="wide")
    inject_styles()
    init_state()
    bundle = train_pipeline()

    st.markdown(
        """
        <div class="hero">
          <h1>ValuEdge | AI House Price Prediction Engine</h1>
          <p>Production-style interactive model with explainable predictions and premium UX.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Property Scenario Presets")
        preset_cols = st.columns(3)
        for i, name in enumerate(SCENARIOS):
            with preset_cols[i]:
                if st.button(name, use_container_width=True):
                    load_scenario(name)
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Property Inputs")
        st.slider(
            "Square Footage",
            min_value=600,
            max_value=4000,
            step=50,
            key="SquareFootage",
            help="Total living area in square feet.",
        )
        st.slider("Bedrooms", min_value=1, max_value=5, key="Bedrooms")
        st.slider("Bathrooms", min_value=1, max_value=4, key="Bathrooms")
        st.slider("Property Age (years)", min_value=1, max_value=60, key="Age")
        st.slider("Garage Spaces", min_value=0, max_value=3, key="GarageSpaces")
        st.slider(
            "Distance to City (km)",
            min_value=1.0,
            max_value=50.0,
            key="DistanceToCity",
            help="Distance from city center in kilometers.",
        )
        predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Model Quality")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{bundle.r2:.4f}")
        m2.metric("Mean Abs Error", f"${bundle.mae:,.0f}")
        m3.metric("Baseline Avg Price", f"${bundle.baseline_price:,.0f}")
        st.write("Top feature influence (coefficient magnitude):")
        ranked = bundle.coef_df.copy()
        ranked["AbsCoefficient"] = ranked["Coefficient"].abs()
        st.bar_chart(ranked.set_index("Feature")["AbsCoefficient"])
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        try:
            inputs = collect_inputs()
            prediction, band = predict_price(bundle, inputs)
            low, high = max(50000.0, prediction - band), prediction + band
            delta_pct = (prediction - bundle.baseline_price) / bundle.baseline_price * 100

            st.markdown("### Prediction Result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Estimated Price", f"${prediction:,.2f}")
            r2.metric("Expected Range", f"${low:,.0f} - ${high:,.0f}")
            r3.metric("Vs Dataset Average", f"{delta_pct:+.1f}%")

            breakdown = pd.DataFrame(
                {
                    "Feature": FEATURE_COLUMNS,
                    "Input Value": [inputs[c] for c in FEATURE_COLUMNS],
                    "Model Coefficient": bundle.model.coef_,
                }
            )
            breakdown["Weighted Impact"] = (
                breakdown["Input Value"] * breakdown["Model Coefficient"]
            )
            st.write("Explainability view (input impact before model intercept scaling):")
            st.dataframe(
                breakdown.sort_values("Weighted Impact", key=np.abs, ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error("Prediction failed. Please verify input values and try again.")
            st.exception(exc)


if __name__ == "__main__":
    render_app()
