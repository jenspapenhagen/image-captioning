from flask import Flask, request, jsonify, Response
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
import requests
import shutil
import re


app = Flask(__name__)

# Example:
# http://127.0.0.1:5000/local?data=test.png
@app.route('/local', methods=['GET', 'POST'])
def local() -> Response:
    if request.method == "GET":
        data: str = request.args.get('data')
        if data is None:
            return jsonify("Please input a valid string: /local?data=test.png")
        print("data ---- > ", data)
        results = predict_step(data)
        return jsonify(results)
    return jsonify("Not a proper request method or data")

# Example:
# http://127.0.0.1:5000/remote?url=https%3A%2F%2Fwww.test.de%2Fimage.png
@app.route('/remote', methods=['GET'])
def remote_dl() -> Response:
    url = request.args.get('url')
    if url is None:
        jsonify("Please input a valid string: /remote?url=https%3A%2F%2Fwww.test.de%2Fimage.png")
    data: list[Image.Image] = loadRemoteImage(url)   
    results = predict_step(data)
    return jsonify(results)


def loadRemoteImage(url: str) -> list[Image.Image]:
    images: list[Image.Image] = []
    fileNameFromURIPattern: str = "/^.*\/(.*)\.(.*)\?.*$/"
    foundFile = re.search(fileNameFromURIPattern, url)
    file_name = foundFile.group()
    if file_name is None:
        raise Exception("Filename missing in URL")
    
    res = requests.get(url, stream = True)

    if res.status_code == 200:
        with open(file_name,'wb') as f:
            shutil.copyfileobj(res.raw, f)
        raise Exception('Image sucessfully Downloaded: ',file_name)
    else:
        raise Exception('Image Couldn\'t be retrieved')
    
    images.append(file_name)
     
    return images

def loadLocalImage(image_path: str) -> list[Image.Image]:
    images: list[Image.Image] = []
    try:        
        loadedImage = Image.open(open(image_path, 'rb'))
        if loadedImage.mode != "RGB":
            loadedImage = loadedImage.convert(mode="RGB")
        images.append(loadedImage)
        if not images:
            raise Exception("can not load image")
        
        return images
    except OSError:
        raise Exception("Image Not Found")


# predict_step(['doctor.e16ba4e4.jpg']) # ['a woman in a hospital bed with a woman in a hospital bed']
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_path: str) -> list:
    realtiv_path = "images/" + image_path
    imagePath = os.path.join(os.getcwd(), realtiv_path)

    images = loadLocalImage(imagePath)
    if not images:
        raise Exception("wrong Image get loaded: " + imagePath)
    

    # build the token out of the image array
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds



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
