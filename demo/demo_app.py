"""
Streamlit Community Cloud entry point -- deliberately separate from app.py,
in its own demo/ subdirectory with its own demo/requirements.txt.

Why a separate directory, not just a separate file: Streamlit Community
Cloud looks for a requirements.txt next to the app's main file first. The
root requirements.txt pins transformers==4.46.3 for a real, Linux-
reproduced DETR-loading bug -- but that pin transitively ceilings
tokenizers<0.21, and building an old tokenizers from source needs a Rust
toolchain new enough for whatever Python Community Cloud currently
defaults to, which isn't guaranteed (confirmed the hard way, twice: torch,
then sentencepiece, then tokenizers-via-transformers all failed to build
against Community Cloud's Python before this split). This directory's
requirements.txt never installs transformers at all, which is safe because
every `transformers` import in src/ (DETR loaders in image_preprocessing.py,
the BLIP loader in models.py) is inside a lazy-loading function body, never
at module level -- and this demo never calls those paths.

What's different from the "real" shipped app, driven by Community Cloud's
free-tier constraints (~1GB RAM; no transformers installed here at all):

1. The ChromaDB index (COCO val2017 + ~12,500 Indian-context images) isn't
   in the repo (chroma_db/ is gitignored) -- downloaded from a public HF
   Dataset on first boot, if not already present locally.
2. Retrieval-only, no crops, no BLIP toggle. The actual shipped default
   (medoid selection + DETR segmentation crops, see src/config.py and the
   main README) needs transformers/DETR and well over 1GB RAM in practice
   (this project's own README documents that from the WSL2 memory-cap
   section) -- neither fits a ~1GB, transformers-free free host. Run this
   project locally via Docker (see the main README) for the real default.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

INDEX_DATASET_REPO = "AntiVicious/rag-image-captioning144-index"
LOCAL_CHROMA_DIR = os.path.join(_REPO_ROOT, "chroma_db")

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

# Retrieval-based captioning has no other notion of "am I right" beyond "how
# close was the nearest thing I found." Empirically checked against four
# real, diverse photos NOT drawn from COCO/the Indian-context set (a casual
# selfie, a terminal screenshot, a stylised anime wallpaper, an outdoor
# chess photo) -- genuinely good matches in this index sit around 0.15-0.35;
# all four out-of-domain probes came back at 0.47-0.86, and their captions
# were confidently wrong (temple/monument captions for a photo of two men
# indoors, a farmers'-protest caption for a chess photo). This index is
# ~37.5K images (COCO val2017 scenes + Indian food/festival/temple photos)
# -- small, and biased toward those two domains. A random personal photo
# has real odds of landing far from anything the index actually contains,
# and the UI should say so instead of presenting a wrong caption as if it
# were a good one.
LOW_CONFIDENCE_DISTANCE = 0.45


@st.cache_resource
def load_pipeline() -> Pipeline:
    config = Config(caption_backend="retrieval", chroma_db_dir=LOCAL_CHROMA_DIR)
    config.enable_segmentation = False
    config.enable_object_detection = False
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
        "(food, festivals, temples). Full methodology, the ablation-backed default this demo "
        "*doesn't* run (medoid selection + DETR segmentation crops -- needs more RAM than this "
        "free host has), and every bug caught along the way: "
        "[github.com/AntiVicious/rag-image-captioning](https://github.com/AntiVicious/rag-image-captioning)."
    )
    st.info(
        "This is the retrieval-only path (no DETR crops, no BLIP) -- lightweight enough for a "
        "free ~1GB host. The actual best-scoring config needs more RAM; run this repo locally "
        "via Docker for that.",
        icon="ℹ️",
    )
    st.warning(
        "**This works best on photos similar to what the index actually contains**: everyday "
        "COCO-style scenes (people, animals, vehicles, sports, furniture, food) or Indian "
        "festival/temple/food photos. It's retrieval, not generation -- it can only return text "
        "from a real match already in its ~37,500-image index, so a photo unlike anything in "
        "that index (a selfie, a screenshot, stylised art, an activity COCO doesn't caption "
        "often) will get a confidently wrong caption, not a vague one. The app flags this when "
        "it detects a low-confidence match.",
        icon="⚠️",
    )

    with st.sidebar:
        st.header("Configuration")
        st.caption("retrieval-only -- the only backend installed in this demo, see demo/requirements.txt")
        if st.button("Initialize pipeline", type="primary"):
            with st.spinner("Loading CLIP and the database..."):
                try:
                    load_pipeline().setup()
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
                pipeline = load_pipeline()
                try:
                    result = pipeline.caption_image(image_path, use_advanced=False)
                    st.markdown(f"**Backend:** {result['backend']}")
                    match_distance = result.get("match_distance")
                    if result["generated_caption"]:
                        if match_distance is not None and match_distance > LOW_CONFIDENCE_DISTANCE:
                            st.error(
                                f"⚠️ Low-confidence match (distance {match_distance:.2f} -- typical "
                                f"good matches are under {LOW_CONFIDENCE_DISTANCE:.2f}). This image "
                                "likely isn't well represented in the index; the caption below is "
                                "probably inaccurate, not just imprecise."
                            )
                        st.info(result["generated_caption"])
                        if match_distance is not None:
                            st.caption(f"Match distance: {match_distance:.3f} (lower = more confident)")
                    else:
                        st.warning(empty_caption_message())
                    with st.expander("Retrieved context"):
                        st.write(result.get("retrieved_context", ""))
                except Exception as e:
                    st.error(f"Captioning failed: {e}")


if __name__ == "__main__":
    main()
