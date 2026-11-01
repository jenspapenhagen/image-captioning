#!/bin/sh
set -e
# --preload loads the model once in the master and shares it with the workers via fork
EAGER_LOAD=1 exec gunicorn --workers=2 --preload --timeout=120 --bind=0.0.0.0:5000 api:app
