import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_dataset(seed: int = 42, n_samples: int = 500) -> pd.DataFrame:
    np.random.seed(seed)
    data = pd.DataFrame(
        {
            "SquareFootage": np.random.randint(600, 4000, n_samples),
            "Bedrooms": np.random.randint(1, 6, n_samples),
            "Bathrooms": np.random.randint(1, 5, n_samples),
            "Age": np.random.randint(1, 60, n_samples),
            "GarageSpaces": np.random.randint(0, 4, n_samples),
            "DistanceToCity": np.random.uniform(1, 50, n_samples),
        }
    )
    data["Price"] = (
        150 * data["SquareFootage"]
        + 8000 * data["Bedrooms"]
        + 12000 * data["Bathrooms"]
        - 500 * data["Age"]
        + 7000 * data["GarageSpaces"]
        - 1500 * data["DistanceToCity"]
        + np.random.normal(0, 20000, n_samples)
    ).clip(50000)
    return data


@st.cache_resource
def train_pipeline():
    data = build_dataset()
    x = data.drop(columns=["Price"])
    y = data["Price"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    score = model.score(x_test_scaled, y_test)

    coef_df = pd.DataFrame(
        {"Feature": x.columns, "Coefficient": model.coef_}
    ).sort_values("Coefficient", key=abs, ascending=False)
    return model, scaler, score, coef_df


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction App")
st.caption("Built from your existing Linear Regression project for internship demo.")

model, scaler, r2_score, coef_df = train_pipeline()

left, right = st.columns([1, 1])
with left:
    st.subheader("Property Inputs")
    sqft = st.slider("Square Footage", min_value=600, max_value=4000, value=1800, step=50)
    beds = st.slider("Bedrooms", min_value=1, max_value=5, value=3)
    baths = st.slider("Bathrooms", min_value=1, max_value=4, value=2)
    age = st.slider("Property Age (years)", min_value=1, max_value=60, value=10)
    garage = st.slider("Garage Spaces", min_value=0, max_value=3, value=1)
    dist = st.slider("Distance to City (km)", min_value=1.0, max_value=50.0, value=12.5)

    if st.button("Predict Price", use_container_width=True):
        sample = pd.DataFrame(
            [
                {
                    "SquareFootage": sqft,
                    "Bedrooms": beds,
                    "Bathrooms": baths,
                    "Age": age,
                    "GarageSpaces": garage,
                    "DistanceToCity": dist,
                }
            ]
        )
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)[0]
        st.success(f"Estimated House Price: ${prediction:,.2f}")

with right:
    st.subheader("Model Snapshot")
    st.metric("R² on Test Split", f"{r2_score:.4f}")
    st.write("Most influential features from your trained model:")
    st.dataframe(coef_df, use_container_width=True, hide_index=True)


