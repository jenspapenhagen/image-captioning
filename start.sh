#!/bin/sh
set -e
docker run -it --rm --name pytorch \
  -p 5000:5000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/images:/app/images" \
  mlimage
