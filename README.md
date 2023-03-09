# image-captioning
playing around with image-captioning model and try to make it prod ready.

1. add an python flask http endpoint
2. move this endpoint into a docker image and mount the model and images from the host
3. try to make flask server prod save. with waitress server
    1. adding CORS
    2. adding Prometheus Metrics
4.


Infos:
the model is: nlpconnect/vit-gpt2-image-captioning

this need be downloaded separate and place in the folder modle/transformers/*
