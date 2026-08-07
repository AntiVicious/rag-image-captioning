"""
Final caption generation.

Config.caption_backend selects the strategy:
  "retrieval" -> the aggregated retrieved captions ARE the output (no LLM,
                 no GPU needed — the default, since this project is
                 developed/tested on machines without one)
  "blip"      -> a BLIP captioner generates the final caption, conditioned on
                 the image plus the retrieved context (see Config.blip2_model_name:
                 BLIP (v1) base by default, not BLIP-2 -- BLIP-2 is impractical
                 without a GPU)
"""

from typing import Dict, Union

from PIL import Image

from .config import Config
from .models import ModelManager
from .rag_retrieval import RAGRetriever


class CaptionGenerator:
    """Produces a final caption for an image via the configured backend."""

    def __init__(self, config: Config, model_manager: ModelManager, retriever: RAGRetriever):
        self.config = config
        self.model_manager = model_manager
        self.retriever = retriever

    def generate(self, image: Union[str, Image.Image], top_k: int = None, use_advanced: bool = True) -> Dict:
        retrieval = self.retriever.retrieve(image, top_k=top_k, use_advanced=use_advanced)

        if self.config.caption_backend == "retrieval":
            return {
                "generated_caption": retrieval["aggregated_caption"],
                "retrieved_context": retrieval["aggregated_caption"],
                "match_distance": retrieval.get("match_distance"),
                "backend": "retrieval",
                "variants_processed": retrieval.get("variants_processed"),
                "detected_objects": retrieval.get("detected_objects", []),
            }

        if self.config.caption_backend == "blip":
            return self._generate_with_blip(image, retrieval)

        raise ValueError(f"Unknown caption_backend: {self.config.caption_backend!r}")

    def _generate_with_blip(self, image: Union[str, Image.Image], retrieval: Dict) -> Dict:
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.model_manager.blip2_model is None or self.model_manager.blip2_processor is None:
            self.model_manager.load_blip2_model()

        # BLIP (v1) is a continuation model, not an instruction-follower like
        # BLIP-2/OPT was: given a text prompt it CONTINUES that text rather than
        # answering it, so the retrieved context is injected as a short seed
        # phrase (its first clause) for the caption to continue from, not a
        # full instructional prompt (which BLIP-2's prompting style used).
        context = (retrieval.get("aggregated_caption") or "").strip()
        seed = context.split(".")[0].strip() if context else ""
        prompt = f"a photo related to: {seed}" if seed else None

        import torch

        if prompt:
            inputs = self.model_manager.blip2_processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.model_manager.blip2_processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.model_manager.device)
        if self.model_manager.device == "cuda" and "pixel_values" in inputs:
            # Move to device first, then cast only the float vision tensor.
            # Casting the whole processor output can turn input_ids/
            # attention_mask into floats on some transformers versions and
            # crash the embedding lookup.
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model_manager.blip2_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
                do_sample=False,
            )
        generated_caption = self.model_manager.blip2_processor.decode(
            generated_ids[0], skip_special_tokens=True
        )
        # BLIP echoes the conditioning prompt verbatim as a prefix of its
        # output (it's continuing the text, not replacing it) -- strip it.
        if prompt and generated_caption.lower().startswith(prompt.lower()):
            generated_caption = generated_caption[len(prompt) :].strip()

        return {
            "generated_caption": generated_caption,
            "retrieved_context": retrieval["aggregated_caption"],
            "match_distance": retrieval.get("match_distance"),
            "backend": "blip",
            "variants_processed": retrieval.get("variants_processed"),
            "detected_objects": retrieval.get("detected_objects", []),
        }
