from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image


class modelloader:

    device = None
    model = None
    feature_extractor = None
    tokenizer = None

    def __init__(self):
        print("laod the local model")
        model_path: str = './models/transformers/'

        self.model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_path, local_files_only=True)
        print("transformer model loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print("transformer tokenizer loaded")

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model.to(self.device)
        print("model competed loaded")



    def predict_step(self, image_list: list[Image.Image]) -> list[str]:
        # build the token out of the image list
        pixel_values = self.feature_extractor(images=image_list, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # predict_step(['pexels-photo-5596193.jpeg']) # ['a gray and white cat sitting on top of a table']
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds