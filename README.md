# image-captioning
playing around with image-captioning model and try to make it prod ready.

1. add a python flask http endpoint
2. move this endpoint into a docker image and mount the model and images from the host
3. try to make flask server prod save. with waitress server
    1. adding CORS
    2. adding Prometheus Metrics
4. loading the model in an extra service for better separation
5. testing other servicer than waitress
6.


Infos:
the model is: [https://huggingface.co/nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

this need be downloaded separate and place in the folder modle/transformers/*

Endpoints:
- `GET /api/local?data=pexels-photo-5596193.jpeg` (only files inside `images/`)
- `GET /api/remote?url=https%3A%2F%2Fexample.com%2Fimage.png` (http/https, public hosts only, 5s timeout, 10 MB limit)
- `GET /healthz` -> health check

Run:
```bash
./build.sh
./start.sh
EAGER_LOAD=1 ./start-gunicore.sh
python3 -m unittest discover -s tests -v
```

Env: 
- `MODEL_PATH` (default `./models/transformers`)
- `EAGER_LOAD=1` loads the model before the first request.

