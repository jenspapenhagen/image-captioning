# image-captioning
playing around with image-captioning model and try to make it prod ready.

first idea: as an python flask http endpoint
sec. : move this endpoint into a docker image and mount the model and images from the host
3. :  try to make falsk prod save.


the model is:
nlpconnect/vit-gpt2-image-captioning

need be downloaded separate
and place in the folder modle/transformers/*