import os
import threading

from PIL import Image

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "transformers")


class ModelLoader:
    """Loads the captioning model once and reuses it for every request."""

    def __init__(self, model_path: str | None = None):
        # imported here so the api module can be imported without torch installed
        import torch
        from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

        self.model_path: str = model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        self.device = torch.device("cpu")

        print(f"load the local model from {self.model_path}")
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path, local_files_only=True)
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_path, local_files_only=True)
        print("transformer model loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        print("transformer tokenizer loaded")

        self.model.to(self.device)
        self.model.eval()
        print("model completely loaded")

    def predict_step(self, image_list: list[Image.Image]) -> list[str]:
        import torch

        # build the token out of the image list
        pixel_values = self.feature_extractor(images=image_list, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        with torch.no_grad():
            output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [pred.strip() for pred in preds]


_instance: ModelLoader | None = None
_lock = threading.Lock()


def get_model() -> ModelLoader:
    """Return the process wide model instance, loading it on first use."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ModelLoader()
    return _instance
