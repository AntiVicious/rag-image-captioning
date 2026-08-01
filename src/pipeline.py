"""
Pipeline: wires Config + ModelManager + DatabaseManager + ImagePreprocessor
+ RAGRetriever + CaptionGenerator into the operations the CLI exposes.
"""
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from PIL import Image

from .config import Config
from .database import DatabaseManager
from .generation import CaptionGenerator
from .image_preprocessing import ImagePreprocessor
from .models import ModelManager
from .rag_retrieval import RAGRetriever


class Pipeline:
    """End-to-end RAG image captioning pipeline for a given Config."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model_manager = ModelManager(self.config)
        self.db_manager = DatabaseManager(self.config)
        self.preprocessor = ImagePreprocessor(self.config)
        self.retriever = RAGRetriever(self.config, self.model_manager, self.db_manager, self.preprocessor)
        self.generator = CaptionGenerator(self.config, self.model_manager, self.retriever)

    def setup(self) -> None:
        """Load CLIP (+ BLIP-2 if configured) and open the database."""
        self.config.create_output_dirs()
        self.model_manager.load_clip_model()
        if self.config.caption_backend == "blip":
            self.model_manager.load_blip2_model()
        self.db_manager.initialize()

    def caption_image(self, image: Union[str, Image.Image], top_k: int = None, use_advanced: bool = True) -> Dict:
        return self.generator.generate(image, top_k=top_k, use_advanced=use_advanced)

    def build_database_from_coco(self) -> bool:
        """Populate the database from the COCO captions dataset."""
        if not self.config.validate_paths():
            return False

        if self.model_manager.clip_model is None:
            self.model_manager.load_clip_model()

        import json

        from tqdm import tqdm

        with open(self.config.coco_ann_file, "r") as f:
            coco_data = json.load(f)

        annotations = coco_data["annotations"]
        img_id_to_path = {
            int(fn.split(".")[0]): os.path.join(self.config.coco_img_dir, fn)
            for fn in os.listdir(self.config.coco_img_dir)
            if fn.endswith(".jpg")
        }

        # COCO has ~5 captions per image. Group by image_id, encode each
        # image ONCE, then emit one row per caption sharing that embedding
        # (instead of re-encoding the same image once per caption).
        captions_by_img: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for ann in annotations:
            captions_by_img[ann["image_id"]].append((ann["id"], ann["caption"]))

        batch_embeddings, batch_documents, batch_ids = [], [], []
        for img_id, ann_list in tqdm(captions_by_img.items()):
            img_path = img_id_to_path.get(img_id)
            if not img_path or not os.path.exists(img_path):
                continue

            tensor = self.preprocessor.preprocess_for_clip(img_path, self.model_manager.clip_preprocess)
            tensor = tensor.to(self.model_manager.device)
            embedding = self.model_manager.encode_image(tensor).cpu().numpy()[0]

            for ann_id, caption in ann_list:
                batch_embeddings.append(embedding)
                batch_documents.append(caption)
                batch_ids.append(f"{img_id}_{ann_id}")

            if len(batch_embeddings) >= self.config.embedding_batch_size:
                self.db_manager.add_embeddings(batch_embeddings, batch_documents, batch_ids)
                batch_embeddings, batch_documents, batch_ids = [], [], []

        if batch_embeddings:
            self.db_manager.add_embeddings(batch_embeddings, batch_documents, batch_ids)

        return True
