"""
Throwaway diagnostic: shows BLIP-2's raw generation output before our
prefix-stripping logic touches it. Run with your own image:

    python diagnose_blip2.py path/to/image.jpg
"""

import sys

from PIL import Image

from src.config import Config
from src.models import ModelManager

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")

config = Config(caption_backend="blip")
mm = ModelManager(config)
print("Loading BLIP-2 (should be fast, already cached)...")
mm.load_blip2_model()

prompt = (
    "Generate a natural, descriptive caption for this image.\n\n"
    "Retrieved captions from similar images: \n\n"
    "Based on what you see in the image and the retrieved context above, "
    "generate a concise caption:"
)

import torch  # noqa: E402

inputs = mm.blip2_processor(images=image, text=prompt, return_tensors="pt").to(mm.device)
print("input_ids length:", inputs["input_ids"].shape[1])

with torch.no_grad():
    generated_ids = mm.blip2_model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        num_beams=config.num_beams,
        do_sample=False,
    )

print("generated_ids shape:", generated_ids.shape)
print("RAW decoded (with prompt if echoed):")
print(repr(mm.blip2_processor.decode(generated_ids[0], skip_special_tokens=True)))

# Also try WITHOUT our custom prompt, using BLIP-2's own documented simple style.
print("\n--- Now trying BLIP-2's own simple prompting style (no custom meta-prompt) ---")
inputs2 = mm.blip2_processor(images=image, return_tensors="pt").to(mm.device)
with torch.no_grad():
    generated_ids2 = mm.blip2_model.generate(**inputs2, max_new_tokens=50)
print("RAW decoded (unconditional, no text prompt):")
print(repr(mm.blip2_processor.decode(generated_ids2[0], skip_special_tokens=True)))
