from flask import Flask, request, jsonify, Response
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os

app = Flask(__name__)


# Example:
# http://127.0.0.1:5000/?data=test.png
@app.route('/', methods=['GET', 'POST'])
def makeCalc():
    if request.method == "GET":
        data: str = request.args.get('data')
        if data is None:
            return "Please input a valid string: ?data=test.png"
        print("data ---- > ", data)
        # predict_step(['doctor.e16ba4e4.jpg']) # ['a woman in a hospital bed with a woman in a hospital bed']
        results = predict_step(data)

        return jsonify(results)
    return "Not a proper request method or data"


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image_path: str) -> str | list:
    realtiv_path = "images/" + image_path
    imagePath = os.path.join(os.getcwd(), realtiv_path)

    images = loadImage(imagePath)
    if not images:
        return "wrong Image get loaded: " + imagePath
    

    # build the token out of the image array
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


def loadImage(image_path: str) -> list[Image.Image] | str:
    images: list[Image.Image] = []
    try:        
        loadedImage = Image.open(open(image_path, 'rb'))
        if loadedImage.mode != "RGB":
            loadedImage = loadedImage.convert(mode="RGB")
        images.append(loadedImage)
        if not images:
            return "can not load image" 
        
        return images
    except OSError:
        return "Image Not Found"


if __name__ == '__main__':
    model_path: str = './models/transformers/'
    model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
    feature_extractor = ViTImageProcessor.from_pretrained(model_path, local_files_only=True)

    print("----------- transformer model loaded ------------")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("----------- transformer tokenizer loaded ------------")

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    print("----------- model loaded ------------")

    app.run(debug=True, host='0.0.0.0')
