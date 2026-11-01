import io
import ipaddress
import logging
import os
import socket
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from flask import Flask, Response, abort, jsonify, request
from flask_cors import CORS
from PIL import Image
from prometheus_flask_exporter import PrometheusMetrics

import modelloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGES_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
REMOTE_TIMEOUT_SECONDS = 5
ALLOWED_REMOTE_SCHEMES = ("http", "https")
METRICS_PATH = "/metrics"

# 40 MP is far above anything the model can process, but stops decompression bombs
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"
metrics = PrometheusMetrics(app, group_by="endpoint")


class ImageLoadError(Exception):
    """Raised when an image can not be loaded from disk or from a URL."""


class BadRequestError(Exception):
    """Raised when user input is rejected."""


# HOWTO: https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask
# Prometheus
# https://blog.viktoradam.net/2020/05/11/prometheus-flask-exporter/
# request duration metrics and request counters exposed on the /metrics endpoint of the Flask application
# Flask CORS
# https://flask-cors.corydolphin.com/en/latest/api.html#decorator

common_counter = metrics.counter(
    "cnt_collection", "Number of invocations per collection",
    labels={
        "collection": lambda: "local",
        "status": lambda resp: resp.status_code,
    },
)
remote_counter = metrics.counter(
    "by_endpoint_counter", "Request count by endpoints",
    labels={"endpoint": lambda: request.endpoint},
)


@app.before_request
def _protect_metrics():
    """Keep /metrics reachable for internal scrapers only."""
    if request.path != METRICS_PATH:
        return None
    if not _is_trusted_client(request.remote_addr):
        abort(403)
    return None


if os.environ.get("EAGER_LOAD", "0") == "1":
    # load before the first request so a broken model volume fails at startup
    modelloader.get_model()


def _get_endpoint() -> modelloader.ModelLoader:
    return modelloader.get_model()


def _predict(image_list: list[Image.Image]) -> list[str]:
    return _get_endpoint().predict_step(image_list)


@app.route("/healthz", methods=["GET"])
def healthz() -> Response:
    return jsonify({"status": "ok"})


# Example:
# http://127.0.0.1:5000/api/local?data=test.png
@app.route("/api/local", methods=["GET"])
@common_counter
def local() -> Response:
    data: str | None = request.args.get("data")
    if not data:
        logger.info("missing query parameter data")
        abort(400, description="Please input a valid string: /api/local?data=test.png")

    try:
        image_path = _resolve_local_path(data)
    except BadRequestError as exc:
        abort(400, description=str(exc))

    if not os.path.isfile(image_path):
        logger.info("image not found: %s", image_path)
        abort(404, description="Image Not Found")

    try:
        images = _load_local_image(image_path)
    except BadRequestError as exc:
        abort(400, description=str(exc))
    except ImageLoadError:
        abort(404, description="Image Not Found")

    return jsonify(_predict(images))


# Example:
# http://127.0.0.1:5000/api/remote?url=https%3A%2F%2Fwww.test.de%2Fimage.png
@app.route("/api/remote", methods=["GET"])
@remote_counter
def remote() -> Response:
    url_parameter: str | None = request.args.get("url")
    if not url_parameter:
        logger.info("missing query parameter url")
        abort(400, description="Please input a valid string: /api/remote?url=https%3A%2F%2Fwww.test.de%2Fimage.png")

    try:
        images = _load_remote_image(url_parameter)
    except BadRequestError as exc:
        abort(400, description=str(exc))
    except ImageLoadError:
        abort(404, description="Image Not Found")

    return jsonify(_predict(images))


@app.errorhandler(500)
def server_error(e):
    logger.exception("An error occurred during a request. %s", e)
    return "An internal error occurred", 500


def _resolve_local_path(data: str) -> str:
    """Resolve a caller supplied file name inside IMAGES_DIR, rejecting traversal."""
    candidate = os.path.realpath(os.path.join(IMAGES_DIR, data))
    try:
        inside = os.path.commonpath([candidate, IMAGES_DIR]) == IMAGES_DIR
    except ValueError:
        inside = False
    if not inside:
        logger.warning("path traversal attempt rejected: %s", data)
        raise BadRequestError("Invalid file name")
    return candidate


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _is_trusted_client(remote_addr: str | None) -> bool:
    try:
        return not _is_public_ip(ipaddress.ip_address(remote_addr or ""))
    except ValueError:
        return False


def _validate_remote_url(url: str) -> None:
    """Reject non http(s) URLs and hosts resolving to internal addresses (SSRF)."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_REMOTE_SCHEMES:
        raise BadRequestError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise BadRequestError("URL has no host")
    if parsed.username or parsed.password:
        raise BadRequestError("Credentials in URLs are not allowed")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BadRequestError("URL host can not be resolved") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not _is_public_ip(ip):
            logger.warning("SSRF attempt rejected: %s -> %s", url, ip)
            raise BadRequestError("URL resolves to a non public address")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Revalidates every redirect target, otherwise a public URL could bounce to localhost."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_remote_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirectHandler)


def _load_remote_image(url: str) -> list[Image.Image]:
    _validate_remote_url(url)

    try:
        with _opener.open(url, timeout=REMOTE_TIMEOUT_SECONDS) as response:
            declared_length = response.headers.get("Content-Length")
            if declared_length and int(declared_length) > MAX_IMAGE_BYTES:
                raise BadRequestError("Image is too large")
            raw = response.read(MAX_IMAGE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        logger.info("remote image could not be downloaded: %s (%s)", url, exc)
        raise ImageLoadError("can not load image") from exc

    if len(raw) > MAX_IMAGE_BYTES:
        raise BadRequestError("Image is too large")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except OSError as exc:
        logger.info("remote content is not an image: %s", url)
        raise ImageLoadError("can not load image") from exc
    except Image.DecompressionBombError as exc:
        logger.warning("decompression bomb rejected: %s", url)
        raise BadRequestError("Image is too large") from exc

    return [image]


def _load_local_image(image_path: str) -> list[Image.Image]:
    try:
        with open(image_path, "rb") as handle:
            image = Image.open(handle).convert("RGB")
    except OSError as exc:
        logger.info("local image could not be opened: %s", image_path)
        raise ImageLoadError("can not load image") from exc
    except Image.DecompressionBombError as exc:
        logger.warning("decompression bomb rejected: %s", image_path)
        raise BadRequestError("Image is too large") from exc

    return [image]


if __name__ == "__main__":
    modelloader.get_model()
    app.run(debug=False, host="0.0.0.0", threaded=True)
