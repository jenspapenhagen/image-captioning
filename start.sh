#!/bin/sh
set -e
docker run -it --rm --name pytorch \
  -p 5000:5000 \
  --read-only \
  --tmpfs /tmp \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/images:/app/images:ro" \
  mlimage
