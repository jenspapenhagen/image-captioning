# image-captioning
playing around with image-captioning model and try to make it prod ready.

1. add an python flask http endpoint
2. move this endpoint into a docker image and mount the model and images from the host
3. try to make flask server prod save. with waitress server
    1. adding CORS
    2. adding Prometheus Metrics
4. loading the model in an extzra service for better seperation
5. testing other servicer than waitress
6.


Infos:
the model is: [https://huggingface.co/nlpconnect/vit-gpt2-image-captioning](nlpconnect/vit-gpt2-image-captioning)

this need be downloaded separate and place in the folder modle/transformers/*

Endpoints:
- `GET /healthz` -> `{"status": "ok"}`
- `GET /api/local?data=pexels-photo-5596193.jpeg` (only files inside `images/`)
- `GET /api/remote?url=https%3A%2F%2Fexample.com%2Fimage.png` (http/https, public hosts only, 5s timeout, 10 MB limit)

Run:
```sh
./build.sh && ./start.sh                    # docker, model+images mounted from the host
EAGER_LOAD=1 ./start-gunicore.sh            # gunicorn
python3 -m unittest discover -s tests -v    # tests
```

Env: `MODEL_PATH` (default `./models/transformers`), `EAGER_LOAD=1` loads the model before the first request.
