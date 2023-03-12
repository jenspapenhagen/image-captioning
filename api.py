from flask import Flask, request, jsonify, Response, abort
from flask_cors import CORS, cross_origin
from prometheus_flask_exporter import PrometheusMetrics
from urllib.request import urlopen
from PIL import Image
import os
import datetime
import logging
import modelloader


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
# extra metric for the remote entpoint
remote_counter = metrics.counter(
    'by_endpoint_counter', 'Request count by endpoints',
    labels={'endpoint': lambda: request.endpoint}
)

# Example:
# http://127.0.0.1:5000/api/local?data=test.png
@app.route('/api/local', methods=['GET'])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
@common_counter
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
        
    results = _predicht(images)

    return jsonify(results)   


# Example:
# http://127.0.0.1:5000/api/remote?url=https%3A%2F%2Fwww.test.de%2Fimage.png
@app.route('/api/remote', methods=['GET'])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
@remote_counter
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

    results = _predicht(images)
    
    return jsonify(results)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request. %s', e)
    return "An internal error occurred", 500

def _predicht( image_list: list[Image.Image]) -> list[str]:
    endpoint = modelloader.modelloader()
    result = endpoint.predict_step(image_list);
    return result;

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

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', threaded=True)
