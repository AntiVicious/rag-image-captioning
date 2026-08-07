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

# Retrieval-based captioning has no notion of "am I right" beyond "how close
# was the nearest match" -- see Result.match_distance (src/rag_retrieval.py).
# Empirically, genuinely good matches against the shipped index sit around
# 0.15-0.35; real out-of-domain probes (a selfie, a screenshot, stylised
# art, an activity photo) came back at 0.47-0.86 with confidently wrong
# captions, not just imprecise ones. Surface that instead of hiding it.
LOW_CONFIDENCE_DISTANCE = 0.45


@st.cache_resource
def load_pipeline(caption_backend: str) -> Pipeline:
    return Pipeline(Config(caption_backend=caption_backend))


def empty_caption_message() -> str:
    """Guidance shown when generation returns an empty caption.

    An empty result almost always means the ChromaDB collection has no
    captions indexed yet (a fresh clone never ran `build-db`), not that
    something crashed -- so this is surfaced as guidance, not an error.
    """
    return (
        "No caption was generated. If this is a fresh clone, the ChromaDB "
        "collection likely has no captions indexed yet. Run "
        "`python -m src.cli build-db` (needs the COCO dataset downloaded "
        "first) to populate it, then try again."
    )


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
        "optionally refined by BLIP."
    )

    with st.sidebar:
        st.header("Configuration")
        backend = st.selectbox(
            "Caption backend",
            ["retrieval", "blip"],
            index=0,
            help="'retrieval' (default) is the ablation-backed winner -- no LLM, no GPU. "
            "'blip' additionally runs a small BLIP captioner conditioned on the retrieved "
            "context; CPU-feasible, but not benchmarked in the ablation.",
        )
        use_advanced = st.checkbox(
            "Use segmentation/object-detection crops",
            value=True,
            help="On by default: under the default consensus (medoid) caption-selection "
            "strategy, segmentation crops score best of every config measured -- see "
            "results/selection_strategy_ablation.csv. Costs real latency (~10s/image on CPU "
            "for DETR); turn off for speed.",
        )

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
                    match_distance = result.get("match_distance")
                    if result["generated_caption"]:
                        if match_distance is not None and match_distance > LOW_CONFIDENCE_DISTANCE:
                            st.error(
                                f"⚠️ Low-confidence match (distance {match_distance:.2f}). Retrieval-"
                                "based captioning can only return text from a real match already in "
                                "the index -- this image likely isn't well represented in it, so the "
                                "caption below may be confidently wrong, not just imprecise."
                            )
                        st.info(result["generated_caption"])
                        if match_distance is not None:
                            st.caption(f"Match distance: {match_distance:.3f} (lower = more confident)")
                    else:
                        st.warning(empty_caption_message())
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
