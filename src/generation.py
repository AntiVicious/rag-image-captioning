"""
Final caption generation.

Config.caption_backend selects the strategy:
  "retrieval" -> the aggregated retrieved captions ARE the output (no LLM,
                 no GPU needed — the default, since this project is
                 developed/tested on machines without one)
  "blip"      -> BLIP-2 generates the final caption, conditioned on the
                 image plus the retrieved context as a text prompt
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

        prompt = (
            "Generate a natural, descriptive caption for this image.\n\n"
            f"Retrieved captions from similar images: {retrieval['aggregated_caption']}\n\n"
            "Based on what you see in the image and the retrieved context above, "
            "generate a concise caption:"
        )

        import torch

        inputs = self.model_manager.blip2_processor(images=image, text=prompt, return_tensors="pt")
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
        generated_caption = self.model_manager.blip2_processor.decode(generated_ids[0], skip_special_tokens=True)
        # Case-insensitive strip of a "caption:" prefix BLIP-2 sometimes
        # echoes back. Search on the lowercased string but slice the
        # original so casing elsewhere in the text is preserved.
        prefix_index = generated_caption.lower().find("caption:")
        if prefix_index != -1:
            generated_caption = generated_caption[prefix_index + len("caption:") :].strip()

        return {
            "generated_caption": generated_caption,
            "retrieved_context": retrieval["aggregated_caption"],
            "backend": "blip",
            "variants_processed": retrieval.get("variants_processed"),
            "detected_objects": retrieval.get("detected_objects", []),
        }
