"""
Streamlit UI for the RAG Image Captioning pipeline.
"""

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import Config
from src.pipeline import Pipeline

st.set_page_config(page_title="RAG Image Captioning", page_icon="🖼️", layout="wide")


@st.cache_resource
def load_pipeline(caption_backend: str) -> Pipeline:
    return Pipeline(Config(caption_backend=caption_backend))


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def main() -> None:
    st.title("🖼️ RAG Image Captioning")
    st.markdown(
        "Upload an image to generate a caption using CLIP + ChromaDB retrieval, "
        "optionally refined by BLIP-2."
    )

    with st.sidebar:
        st.header("Configuration")
        backend = st.selectbox(
            "Caption backend",
            ["retrieval", "blip"],
            index=0,
            help="'retrieval' needs no GPU. 'blip' generates via BLIP-2 and needs GPU weights.",
        )
        use_advanced = st.checkbox("Use segmentation/object-detection crops", value=True)

        if st.button("Initialize pipeline", type="primary"):
            with st.spinner("Loading models and database..."):
                try:
                    load_pipeline(backend).setup()
                    st.session_state["initialized"] = True
                    st.success("Pipeline initialized.")
                except Exception as e:
                    st.session_state["initialized"] = False
                    st.error(f"Initialization failed: {e}")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    image_path = save_uploaded_file(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded image")
        st.image(Image.open(image_path), width="stretch")

    with col2:
        st.subheader("Result")
        if not st.session_state.get("initialized", False):
            st.warning("Initialize the pipeline in the sidebar first.")
            return

        if st.button("Generate caption", type="primary"):
            with st.spinner("Captioning..."):
                pipeline = load_pipeline(backend)
                try:
                    result = pipeline.caption_image(image_path, use_advanced=use_advanced)
                    st.markdown(f"**Backend:** {result['backend']}")
                    st.info(result["generated_caption"])
                    with st.expander("Retrieved context"):
                        st.write(result.get("retrieved_context", ""))
                    if result.get("detected_objects"):
                        st.markdown("**Detected objects:**")
                        for obj in result["detected_objects"]:
                            st.write(f"- label {obj['label_id']} (confidence {obj['score']:.2%})")
                except Exception as e:
                    st.error(f"Captioning failed: {e}")


if __name__ == "__main__":
    main()
