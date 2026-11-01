import io
import os
import socket
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api
import modelloader

PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeModel:
    def predict_step(self, image_list):
        return [f"caption for {len(image_list)} image(s)"]


class ApiTest(unittest.TestCase):
    def setUp(self):
        modelloader._instance = FakeModel()
        self.client = api.app.test_client()

    def tearDown(self):
        modelloader._instance = None

    def test_healthz(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)

    def test_local_ok(self):
        response = self.client.get("/api/local?data=pexels-photo-5596193.jpeg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), ["caption for 1 image(s)"])

    def test_local_missing_param(self):
        self.assertEqual(self.client.get("/api/local").status_code, 400)

    def test_local_unknown_file(self):
        self.assertEqual(self.client.get("/api/local?data=nope.png").status_code, 404)

    def test_local_path_traversal(self):
        for payload in ("../api.py", "../../etc/passwd", "/etc/passwd"):
            response = self.client.get("/api/local", query_string={"data": payload})
            self.assertEqual(response.status_code, 400, payload)

    def test_remote_rejects_ssrf(self):
        for url in ("http://127.0.0.1/x.png", "http://localhost/x.png", "file:///etc/passwd",
                    "http://10.0.0.1/x.png", "http://169.254.169.254/latest/meta-data"):
            response = self.client.get("/api/remote", query_string={"url": url})
            self.assertEqual(response.status_code, 400, url)

    def test_remote_ok(self):
        fake_response = mock.MagicMock()
        fake_response.headers = {"Content-Length": str(len(PNG))}
        fake_response.read.return_value = PNG
        fake_response.__enter__.return_value = fake_response

        with mock.patch.object(api, "_opener") as opener, \
                mock.patch.object(api.socket, "getaddrinfo",
                                  return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]):
            opener.open.return_value = fake_response
            response = self.client.get("/api/remote", query_string={"url": "https://example.com/x.png"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), ["caption for 1 image(s)"])

    def test_remote_too_large(self):
        fake_response = mock.MagicMock()
        fake_response.headers = {"Content-Length": str(api.MAX_IMAGE_BYTES + 1)}
        fake_response.read.return_value = b""
        fake_response.__enter__.return_value = fake_response

        with mock.patch.object(api, "_opener") as opener, \
                mock.patch.object(api.socket, "getaddrinfo",
                                  return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]):
            opener.open.return_value = fake_response
            response = self.client.get("/api/remote", query_string={"url": "https://example.com/x.png"})

        self.assertEqual(response.status_code, 400)

    def test_remote_invalid_image(self):
        fake_response = mock.MagicMock()
        fake_response.headers = {}
        fake_response.read.return_value = b"not an image"
        fake_response.__enter__.return_value = fake_response

        with mock.patch.object(api, "_opener") as opener, \
                mock.patch.object(api.socket, "getaddrinfo",
                                  return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]):
            opener.open.return_value = fake_response
            response = self.client.get("/api/remote", query_string={"url": "https://example.com/x.png"})

        self.assertEqual(response.status_code, 404)

    def test_redirect_revalidation(self):
        handler = api._SafeRedirectHandler()
        with self.assertRaises(api.BadRequestError):
            handler.redirect_request(mock.MagicMock(), None, 302, "Found", {}, "http://127.0.0.1/x.png")

    def test_decompression_bomb_rejected(self):
        with mock.patch.object(api.Image, "MAX_IMAGE_PIXELS", 0):
            response = self.client.get("/api/local?data=pexels-photo-5596193.jpeg")
        self.assertEqual(response.status_code, 400)

    def test_metrics_allowed_from_private_client(self):
        self.assertEqual(self.client.get("/metrics").status_code, 200)

    def test_metrics_blocked_from_public_client(self):
        response = self.client.get("/metrics", environ_base={"REMOTE_ADDR": "93.184.216.34"})
        self.assertEqual(response.status_code, 403)



if __name__ == "__main__":
    unittest.main()
