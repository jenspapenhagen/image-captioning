from flask import Flask, request, jsonify, Response, abort, logging
from flask_cors import CORS, cross_origin
from prometheus_flask_exporter import PrometheusMetrics
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from urllib.request import urlopen
from PIL import Image
import os
import datetime

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['CORS_HEADERS'] = 'Content-Type'
app.permanent_session_lifetime = datetime.timedelta(days=365)
metrics = PrometheusMetrics(app, group_by='endpoint')


# HOWTO: https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask
# Prometheus
# https://blog.viktoradam.net/2020/05/11/prometheus-flask-exporter/
# request duration metrics and request counters exposed on the /metrics endpoint of the Flask application
# Flask CORS
# https://flask-cors.corydolphin.com/en/latest/api.html#decorator

# custom metric to be applied to multiple endpoints
common_counter = metrics.counter(
  'cnt_collection', 'Number of invocations per collection', labels={
        'collection': lambda: request.args.get('data'),
        'status': lambda resp: resp.status_code
    }
)
remote_counter = metrics.counter(
    'by_endpoint_counter', 'Request count by endpoints',
    labels={'endpoint': lambda: request.endpoint}
)

# Example:
# http://127.0.0.1:5000/api/local?data=test.png
@app.route('/api/local', methods=['GET'])
@common_counter
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def local() -> Response:    
    data: str = request.args.get('data')
    if data is None:
        print("Please input a valid string: /local?data=test.png")
        abort(400)
    print("data ---- > ", data)

    realtiv_path = "images/" + data
    image_path = os.path.join(os.getcwd(), realtiv_path)
    images: list[Image.Image] = _load_local_image(image_path)

    if not images:
        print("wrong Image get loaded: " + image_path)
        abort(404)
    results = _predict_step(images)

    return jsonify(results)   


# Example:
# http://127.0.0.1:5000/api/remote?url=https%3A%2F%2Fwww.test.de%2Fimage.png
@app.route('/api/remote', methods=['GET'])
@remote_counter
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def remote() -> Response:
    url_parameter = request.args.get('url')
    if url_parameter is None:
        print("Please input a valid string: /remote?url=https%3A%2F%2Fwww.test.de%2Fimage.png")
        abort(400)
    print("URL ---- > ", url_parameter)

    images: list[Image.Image] = _load_remote_image(url_parameter)
    if not images:
        print("wrong Image get loaded: " + url_parameter)
        abort(404)
    results = _predict_step(images)
    
    return jsonify(results)

# register additional default metrics
metrics.register_default(
    metrics.counter(
        'by_path_counter', 'Request count by request paths',
        labels={'path': lambda: request.path}
    )
)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request. %s', e)
    return "An internal error occurred", 500

def _load_remote_image(url: str) -> list[Image.Image]:
    try:
        images: list[Image.Image] = []
        img = Image.open(urlopen(url))

        images.append(img)
        # check for empty list
        if not images:
            raise Exception("can not load image")

        return images
    except OSError:
        raise Exception("Image Not Found")


def _load_local_image(image_path: str) -> list[Image.Image]:
    try:
        images: list[Image.Image] = []
        loaded_image = Image.open(open(image_path, 'rb'))
        if loaded_image.mode != "RGB":
            loaded_image = loaded_image.convert(mode="RGB")
        images.append(loaded_image)

        # check for empty list
        if not images:
            raise Exception("can not load image")

        return images
    except OSError:
        raise Exception("Image Not Found")


# predict_step(['pexels-photo-5596193.jpeg']) # ['a gray and white cat sitting on top of a table']
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def _predict_step(image_list: list[Image.Image]) -> list[str]:
    # build the token out of the image list
    pixel_values = feature_extractor(images=image_list, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


if __name__ == '__main__':
    print("laod the local model")
    model_path: str = './models/transformers/'

    model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
    feature_extractor = ViTImageProcessor.from_pretrained(model_path, local_files_only=True)
    print("transformer model loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("transformer tokenizer loaded")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    print("model competed loaded")

    app.run(debug=False, host='0.0.0.0', threaded=True)