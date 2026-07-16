import os
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 AI Image Generator")
st.write("Generate beautiful AI images using FLUX")

prompt = st.text_area("Enter your prompt")

if st.button("Generate Image"):

    if not HF_TOKEN:
        st.error("HF_API_TOKEN not found in .env")
        st.stop()

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
        st.stop()

    try:
        client = InferenceClient(
            api_key=HF_TOKEN
        )

        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-dev"
        )

        st.image(image, caption=prompt)

    except Exception as e:
        st.error(str(e))