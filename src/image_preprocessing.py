"""
Advanced image preprocessing: segmentation and object-detection crops.

Detection models are lazy-loaded on first use (not at import time, and not
at ImagePreprocessor construction), so importing this module or building a
preprocessor never triggers a multi-hundred-MB weights download.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .config import Config


class ImagePreprocessor:
    """Generates segmentation/object-detection crops for a Config."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.segmentation_processor = None
        self.segmentation_model = None
        self.object_detection_processor = None
        self.object_detection_model = None

    def _ensure_segmentation_model(self) -> None:
        if self.segmentation_model is not None:
            return
        from transformers import DetrForSegmentation, DetrImageProcessor

        self.segmentation_processor = DetrImageProcessor.from_pretrained(self.config.segmentation_model)
        self.segmentation_model = DetrForSegmentation.from_pretrained(self.config.segmentation_model)
        self.segmentation_model.to(self.device)
        self.segmentation_model.eval()

    def _ensure_object_detection_model(self) -> None:
        if self.object_detection_model is not None:
            return
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self.object_detection_processor = DetrImageProcessor.from_pretrained(self.config.object_detection_model)
        self.object_detection_model = DetrForObjectDetection.from_pretrained(self.config.object_detection_model)
        self.object_detection_model.to(self.device)
        self.object_detection_model.eval()

    def preprocess_for_clip(self, image: Union[str, Image.Image], clip_preprocess):
        """Load (if needed), RGB-convert, and run an image through CLIP's transform."""
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return clip_preprocess(image).unsqueeze(0)

    def get_segmentation_crops(self, image: Image.Image) -> List[Image.Image]:
        """Crop out the top-N largest semantic segments (panoptic segmentation)."""
        if not self.config.enable_segmentation:
            return []
        self._ensure_segmentation_model()

        import torch

        inputs = self.segmentation_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.segmentation_model(**inputs)

        # DetrImageProcessor's panoptic checkpoint exposes
        # post_process_panoptic_segmentation (not post_process_segmentation,
        # which doesn't exist on this class).
        result = self.segmentation_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        segmentation = result.get("segmentation")
        if segmentation is None:
            return []

        segment_sizes = []
        for seg_id in torch.unique(segmentation):
            # 0 / negative ids are background/void in panoptic segmentation.
            if seg_id.item() <= 0:
                continue
            size = (segmentation == seg_id).sum().item()
            if size < 100:  # ignore tiny noise segments
                continue
            segment_sizes.append((seg_id, size))
        segment_sizes.sort(key=lambda item: item[1], reverse=True)

        crops = []
        for seg_id, _ in segment_sizes[: self.config.crops_per_mode]:
            crop = self._crop_from_mask(image, segmentation == seg_id)
            if crop is not None:
                crops.append(crop)
        return crops

    def get_object_detection_crops(self, image: Image.Image) -> Tuple[List[Image.Image], List[Dict]]:
        """Crop out the top-N highest-confidence detected objects."""
        if not self.config.enable_object_detection:
            return [], []
        self._ensure_object_detection_model()

        import torch

        inputs = self.object_detection_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.object_detection_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.object_detection_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.config.detection_score_threshold,
        )[0]

        scores, boxes, labels = results["scores"], results["boxes"], results["labels"]
        top_indices = scores.argsort(descending=True)[: self.config.crops_per_mode]

        crops: List[Image.Image] = []
        detected: List[Dict] = []
        for idx in top_indices:
            box = boxes[idx].cpu().numpy()
            crop = self._crop_from_bbox(image, box)
            if crop is not None:
                crops.append(crop)
                detected.append(
                    {
                        "label_id": labels[idx].item(),
                        "score": scores[idx].item(),
                        "box": box.tolist(),
                    }
                )
        return crops, detected

    def _crop_from_mask(self, image: Image.Image, mask) -> Optional[Image.Image]:
        mask_np = mask.cpu().numpy().astype(np.uint8)
        coords = np.column_stack(np.where(mask_np > 0))
        if len(coords) == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return self._pad_and_crop(image, x_min, y_min, x_max, y_max)

    def _crop_from_bbox(self, image: Image.Image, bbox: np.ndarray) -> Optional[Image.Image]:
        x_min, y_min, x_max, y_max = bbox
        return self._pad_and_crop(image, x_min, y_min, x_max, y_max)

    def _pad_and_crop(self, image: Image.Image, x_min, y_min, x_max, y_max) -> Optional[Image.Image]:
        x_min = max(0, x_min - self.config.crop_padding)
        y_min = max(0, y_min - self.config.crop_padding)
        x_max = min(image.width, x_max + self.config.crop_padding)
        y_max = min(image.height, y_max + self.config.crop_padding)
        if (x_max - x_min) < self.config.min_crop_size or (y_max - y_min) < self.config.min_crop_size:
            return None
        return image.crop((x_min, y_min, x_max, y_max))

    def get_all_variants(self, image: Union[str, Image.Image]) -> Dict:
        """Return the original image plus every configured crop variant."""
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        segmentation_crops = self.get_segmentation_crops(image)
        object_detection_crops, detected_objects = self.get_object_detection_crops(image)

        return {
            "original": image,
            "segmentation_crops": segmentation_crops,
            "object_detection_crops": object_detection_crops,
            "detected_objects": detected_objects,
        }
