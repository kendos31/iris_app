
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os

# Set page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_classifier.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
IMAGE_PATH = os.path.join(BASE_DIR, "iris_flowers.png")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    iris = load_iris()
    return model, scaler, iris

model, scaler, iris = load_model()

# Initialize session state for sliders
defaults = [5.8, 3.0, 4.0, 1.2]
slider_keys = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for k, v in zip(slider_keys, defaults):
    if k not in st.session_state:
        st.session_state[k] = v

# Title and description
st.title("🌸 Iris Flower Classification App")
st.markdown("Adjust the sliders to input flower measurements and see the prediction!")

# Sidebar sliders
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, key='sepal_length')
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, key='sepal_width')
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, key='petal_length')
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, key='petal_width')

# Prepare feature vector
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_features)

# Prediction
prediction = model.predict(input_scaled)[0]
species_name = iris.target_names[prediction]
probabilities = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None

# Display results
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Prediction Result")
    color_map = {'setosa': '#FF6B6B','versicolor': '#4ECDC4','virginica': '#45B7D1'}
    st.markdown(f"<div style='background-color:{color_map[species_name]}; padding: 20px; border-radius: 10px; text-align:center;'><h2 style='color:white;'>{species_name.upper()}</h2></div>", unsafe_allow_html=True)

    if probabilities is not None:
        st.subheader("Prediction Confidence")
        for s, p in zip(iris.target_names, probabilities):
            st.write(f"**{s.capitalize()}**: {p*100:.1f}%")
            st.progress(p)

with col2:
    st.subheader("Your Input Measurements")
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    feature_values = input_features[0]
    bar_colors = ['#96CEB4' if 'Sepal' in name else color_map[species_name] for name in feature_names]
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(feature_names, feature_values, color=bar_colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel("cm")
    ax.set_ylim(0, max(feature_values)*1.2)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{bar.get_height():.1f}", ha='center')
    plt.tight_layout()
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Feature Importance Demo")

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Display Feature Importance
st.subheader("Feature Importance")
if hasattr(model, 'feature_importances_'):
    fi_df = pd.DataFrame({
        'Feature': iris.feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)  # ascending=True for horizontal bar

    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color='#4ECDC4')
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)
else:
    st.write("Model does not have feature_importances_ attribute.")




# iris_info_app.py
import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Iris Flower Information", layout="wide")
st.title("🌸 Iris Flower Information Viewer")

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Target'] = iris.target
target_mapping = {i: name for i, name in enumerate(iris.target_names)}

# Map target numbers to names
df['Target Name'] = df['Target'].map(target_mapping)

# --- Expander: Dataset Overview ---
with st.expander("📊 Dataset Overview"):
    st.dataframe(df)
    st.write("Feature names:", iris.feature_names)
    st.write("Target classes:", iris.target_names)

# --- Expander: Feature Description ---
feature_desc = {
    "sepal length (cm)": "Length of the sepal in centimeters",
    "sepal width (cm)": "Width of the sepal in centimeters",
    "petal length (cm)": "Length of the petal in centimeters",
    "petal width (cm)": "Width of the petal in centimeters"
}

with st.expander("🌿 Feature Description"):
    for feature, desc in feature_desc.items():
        st.write(f"**{feature}**: {desc}")

# --- Expander: Flower Images ---
from PIL import Image

st.title("Iris Flowers")

# Load an image from file
image = Image.open(IMAGE_PATH)  # replace with your image path

# Display the image
st.image(image, caption="Here is the image", width=1000)




