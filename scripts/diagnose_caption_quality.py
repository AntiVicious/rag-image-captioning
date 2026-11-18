"""Diagnostic: runs the demo pipeline (retrieval-only, medoid selection, no
crops, default_top_k=5) on test images and prints the retrieved candidates
and distances as well as the final caption.

Usage: python scripts/diagnose_caption_quality.py /app/test_images/*.jpg
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.database import DatabaseManager  # noqa: E402
from src.image_preprocessing import ImagePreprocessor  # noqa: E402
from src.models import ModelManager  # noqa: E402

config = Config(caption_backend="retrieval")
config.enable_segmentation = False
config.enable_object_detection = False

model_manager = ModelManager(config)
db_manager = DatabaseManager(config)
preprocessor = ImagePreprocessor(config)

print("Loading CLIP...")
model_manager.load_clip_model()
db_manager.initialize()
print(f"Index: {db_manager.get_stats()['total_embeddings']} embeddings\n")

for image_path in sys.argv[1:]:
    tensor = preprocessor.preprocess_for_clip(image_path, model_manager.clip_preprocess)
    tensor = tensor.to(model_manager.device)
    embedding = model_manager.encode_image(tensor)
    query_embedding = embedding.cpu().numpy()[0].tolist()

    results = db_manager.query_similar([query_embedding], config.default_top_k)
    docs = (results.get("documents") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    print(f"=== {image_path} ===")
    for doc, dist in zip(docs, dists):
        print(f"  dist={dist:.4f}  {doc!r}")
    print()
