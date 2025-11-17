import streamlit as st
import requests
from PIL import Image

st.title("Login")

# Login form
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    # Call your FastAPI token endpoint to validate
    token_url = "https://illatively-crackly-chuck.ngrok-free.dev/token"
    data = {
        "username": username,
        "password": password,
        "grant_type": "password",  # to match OAuth2 form
        "scope": "",
        "client_id": "",
        "client_secret": ""
    }
    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        token = response.json().get("access_token")
        st.session_state.token = token
        st.success("Logged in successfully!")
    else:
        st.error("Login failed! Please check credentials.")

# Show Cat vs Dog classifier only if logged in
if "token" in st.session_state and st.session_state.token:
    st.title("Cat vs Dog Classifier")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
    if uploaded_file:
        if st.button("Predict"):
            files = {"file": uploaded_file.getvalue()}
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            api_url = "https://illatively-crackly-chuck.ngrok-free.dev/predict"
            response = requests.post(api_url, files=files, headers=headers)
            if response.status_code == 200:
                result = response.json()
                st.write(f"Prediction: {result['label']}")
                st.write(f"Confidence: {result['confidence']:.2f}")
            else:
                st.error(f"API error: {response.text}")
