import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="California Home Value Predictor",
    page_icon="🏠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
MODEL_PATH = "rfmodel.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


@st.cache_resource
def build_scaler():
    data = pd.read_csv(DATA_URL)
    data["total_bedrooms"] = data["total_bedrooms"].fillna(
        data["total_bedrooms"].median()
    )
    encoded = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)
    encoded["rooms_per_household"] = encoded["total_rooms"] / encoded["households"]
    encoded["bedrooms_per_room"] = encoded["total_bedrooms"] / encoded["total_rooms"]
    encoded["population_per_household"] = encoded["population"] / encoded["households"]
    features = encoded.drop("median_house_value", axis=1)
    training_features, _ = train_test_split(features, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(training_features)
    return scaler, list(features.columns)


def make_features(values, feature_columns):
    row = pd.DataFrame([values])
    row["rooms_per_household"] = row["total_rooms"] / row["households"]
    row["bedrooms_per_room"] = row["total_bedrooms"] / row["total_rooms"]
    row["population_per_household"] = row["population"] / row["households"]
    row["ocean_proximity_INLAND"] = int(values["ocean_proximity"] == "INLAND")
    row["ocean_proximity_ISLAND"] = int(values["ocean_proximity"] == "ISLAND")
    row["ocean_proximity_NEAR BAY"] = int(values["ocean_proximity"] == "NEAR BAY")
    row["ocean_proximity_NEAR OCEAN"] = int(values["ocean_proximity"] == "NEAR OCEAN")
    row = row.drop(columns=["ocean_proximity"])
    return row.reindex(columns=feature_columns, fill_value=0)


st.markdown("# California Home Value Predictor")
st.caption("A Random Forest model trained on the California housing dataset")

try:
    model = load_model()
    scaler, feature_columns = build_scaler()
except Exception as error:
    st.error(f"The model or training data could not be loaded: {error}")
    st.stop()

with st.sidebar:
    st.header("District details")
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.23, step=0.01)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.88, step=0.01)
    housing_median_age = st.number_input("Housing median age", min_value=1.0, max_value=60.0, value=25.0, step=1.0)
    total_rooms = st.number_input("Total rooms", min_value=1.0, value=2500.0, step=100.0)
    total_bedrooms = st.number_input("Total bedrooms", min_value=1.0, value=500.0, step=50.0)
    population = st.number_input("Population", min_value=1.0, value=1200.0, step=100.0)
    households = st.number_input("Households", min_value=1.0, value=450.0, step=25.0)
    median_income = st.number_input("Median income (in $10,000s)", min_value=0.0, value=4.0, step=0.1)
    ocean_proximity = st.selectbox(
        "Ocean proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    )
    predict = st.button("Predict home value", type="primary", use_container_width=True)

if predict:
    values = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }
    features = make_features(values, feature_columns)
    prediction = float(model.predict(scaler.transform(features))[0])
    st.success(f"Estimated median house value: ${prediction:,.0f}")
    st.metric("Prediction", f"${prediction:,.0f}")
else:
    st.info("Enter district details in the sidebar and select Predict home value.")

st.divider()
st.caption(f"Model: Random Forest Regressor • {len(feature_columns)} processed features")