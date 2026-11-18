"""
Streamlit Community Cloud app, with its own demo/requirements.txt (see
demo/README.md for why it lives in a separate directory).

Differences from app.py, due to the free tier's ~1 GB of RAM:

1. The ChromaDB index is downloaded from a Hugging Face dataset on first
   start instead of being built locally.
2. The upload tab is retrieval-only, without DETR crops or BLIP, which need
   transformers and more memory. The gallery tab shows precomputed results
   for all configurations.
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

import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import Config  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

st.set_page_config(page_title="RAG Image Captioning (demo)", page_icon="🖼️", layout="wide")

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
GALLERY_DATA_PATH = os.path.join(_DEMO_DIR, "gallery_data.json")
GALLERY_IMAGES_DIR = os.path.join(_DEMO_DIR, "gallery_images")

CROP_LABELS = {
    "retrieval-only": "No crops",
    "segmentation": "Segmentation crops",
    "object-detection": "Object-detection crops",
    "all-seven": "Both crop types",
}
SELECTION_LABELS = {"top1": "top1 (nearest-neighbor)", "medoid": "medoid (consensus)"}

# The match distance is the only confidence signal. Good matches in this
# index score about 0.15-0.35. Four out-of-domain test photos (a selfie, a
# screenshot, artwork, a chess game) scored 0.47-0.86 and got wrong captions.
# The index (~37.5K images of COCO scenes and Indian food, festivals and
# temples) is small, so the UI warns when a match is this far.
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


@st.cache_data
def load_gallery_data():
    if not os.path.exists(GALLERY_DATA_PATH):
        return None
    with open(GALLERY_DATA_PATH) as f:
        return json.load(f)


def render_gallery() -> None:
    gallery = load_gallery_data()
    if gallery is None:
        st.warning("Gallery data not found (demo/gallery_data.json is missing).")
        return

    st.markdown(
        "40 held-out COCO val2017 images, run offline through all 8 configs this project's "
        "ablation covers (2 selection strategies x 4 crop configs) — the same measurement "
        "behind the headline finding in the main README. Flip the toggles below and watch the "
        "caption change; nothing here calls a model, it's all precomputed."
    )

    col_toggles, col_result = st.columns([1, 2])

    with col_toggles:
        images = gallery["images"]
        labels = [f"#{e['image_id']}" for e in images]
        idx = st.selectbox("Image", options=range(len(images)), format_func=lambda i: labels[i])
        entry = images[idx]

        selection = st.radio(
            "Selection strategy",
            options=gallery["selection_strategies"],
            format_func=lambda s: SELECTION_LABELS.get(s, s),
            index=1,  # medoid -- the shipped default
        )
        crop_config = st.radio(
            "Crop config",
            options=gallery["crop_configs"],
            format_func=lambda c: CROP_LABELS.get(c, c),
            index=1,  # segmentation -- the shipped default
        )
        if selection == "medoid" and crop_config == "segmentation":
            st.caption("This is the shipped default — the best config measured in the ablation.")

    with col_result:
        img_path = os.path.join(GALLERY_IMAGES_DIR, entry["file"])
        st.image(Image.open(img_path), width="stretch")

        config_key = f"{selection}__{crop_config}"
        result = entry["configs"][config_key]
        st.info(result["caption"])
        if result.get("cider") is not None:
            st.caption(f"CIDEr vs. real COCO references for this image: {result['cider']:.3f}")

        with st.expander("Real COCO reference captions (ground truth, not model output)"):
            for cap in entry["reference_captions"]:
                st.write(f"- {cap}")


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def render_upload_tab() -> None:
    st.info(
        "This is the retrieval-only path (no DETR crops, no BLIP) -- lightweight enough for a "
        "free ~1GB host. The actual best-scoring config needs more RAM; see the gallery tab "
        "for real outputs from that config, or run this repo locally via Docker.",
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
                                "likely isn't well represented in the index, so the caption below is "
                                "probably wrong."
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


def main() -> None:
    st.title("🖼️ RAG Image Captioning — demo")
    st.markdown(
        "CLIP + ChromaDB retrieval over COCO val2017 + ~12,500 Indian-context images "
        "(food, festivals, temples). Full methodology, the ablation-backed default this demo's "
        "upload tab *doesn't* run live (medoid selection + DETR segmentation crops -- needs more "
        "RAM than this free host has, though the gallery tab below shows its real output), and "
        "every bug caught along the way: "
        "[github.com/AntiVicious/rag-image-captioning](https://github.com/AntiVicious/rag-image-captioning)."
    )

    tab_gallery, tab_upload = st.tabs(["📸 Precomputed gallery", "📤 Upload your own"])
    with tab_gallery:
        render_gallery()
    with tab_upload:
        render_upload_tab()


if __name__ == "__main__":
    main()
