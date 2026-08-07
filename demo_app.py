"""
Streamlit Community Cloud entry point -- deliberately separate from app.py.

Two things are different here from the "real" shipped app, both driven by
Community Cloud's free-tier constraints (~1GB RAM, deploys straight from
this public GitHub repo where chroma_db/ is gitignored):

1. The ChromaDB index (COCO val2017 + ~12,500 Indian-context images) isn't
   in this repo -- it's downloaded from a public HF Dataset on first boot,
   if not already present locally.
2. The UI defaults to retrieval-only, crops off. The actual shipped default
   (medoid selection + segmentation crops, see src/config.py and the main
   README) loads CLIP *and* DETR simultaneously, which this project's own
   README already documents as needing well over 1GB RAM on a real machine
   (the WSL2 memory-cap section) -- not a safe default for a ~1GB free
   host. The full config is still reachable via the sidebar toggle; it's
   just not what loads by default here, to keep the demo from crashing.

Everything else (retrieval, augmentation, the ablation-backed defaults) is
identical to app.py -- run this project locally via Docker (see the main
README) to use the real shipped default without the free-tier RAM cap.
"""

import os

INDEX_DATASET_REPO = "AntiVicious/rag-image-captioning144-index"
LOCAL_CHROMA_DIR = "./chroma_db"

if not os.path.exists(LOCAL_CHROMA_DIR) or not os.listdir(LOCAL_CHROMA_DIR):
    import streamlit as st

    with st.spinner(f"First boot: downloading the demo index from {INDEX_DATASET_REPO} (~110MB)..."):
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=INDEX_DATASET_REPO, repo_type="dataset", local_dir=LOCAL_CHROMA_DIR)

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import Config
from src.pipeline import Pipeline

st.set_page_config(page_title="RAG Image Captioning (demo)", page_icon="🖼️", layout="wide")


@st.cache_resource
def load_pipeline(caption_backend: str, enable_segmentation: bool) -> Pipeline:
    config = Config(caption_backend=caption_backend, chroma_db_dir=LOCAL_CHROMA_DIR)
    config.enable_segmentation = enable_segmentation
    return Pipeline(config)


def empty_caption_message() -> str:
    return (
        "No caption was generated. If this is a fresh deploy, the index download above may "
        "not have finished -- refresh and try again."
    )


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def main() -> None:
    st.title("🖼️ RAG Image Captioning — demo")
    st.markdown(
        "CLIP + ChromaDB retrieval over COCO val2017 + ~12,500 Indian-context images "
        "(food, festivals, temples). Full methodology, the ablation behind these defaults, "
        "and every bug caught along the way: "
        "[github.com/AntiVicious/rag-image-captioning](https://github.com/AntiVicious/rag-image-captioning)."
    )
    st.info(
        "This free-tier host has ~1GB RAM, too little to safely run this project's actual "
        "shipped default (medoid selection + DETR segmentation crops, the best-scoring config "
        "measured). Retrieval-only is on by default here for stability; turn on crops below to "
        "try the better-scoring config anyway -- it may be slow or fail on this host's RAM limit.",
        icon="ℹ️",
    )

    with st.sidebar:
        st.header("Configuration")
        backend = st.selectbox("Caption backend", ["retrieval", "blip"], index=0)
        use_advanced = st.checkbox(
            "Use segmentation crops (medoid selection) -- the actual best-scoring config, "
            "risky on ~1GB RAM",
            value=False,
        )

        if st.button("Initialize pipeline", type="primary"):
            with st.spinner("Loading models and database..."):
                try:
                    load_pipeline(backend, use_advanced).setup()
                    st.session_state["initialized"] = True
                    st.session_state["use_advanced"] = use_advanced
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
                pipeline = load_pipeline(backend, st.session_state.get("use_advanced", False))
                try:
                    result = pipeline.caption_image(
                        image_path, use_advanced=st.session_state.get("use_advanced", False)
                    )
                    st.markdown(f"**Backend:** {result['backend']}")
                    if result["generated_caption"]:
                        st.info(result["generated_caption"])
                    else:
                        st.warning(empty_caption_message())
                    with st.expander("Retrieved context"):
                        st.write(result.get("retrieved_context", ""))
                except Exception as e:
                    st.error(f"Captioning failed: {e}")


if __name__ == "__main__":
    main()
