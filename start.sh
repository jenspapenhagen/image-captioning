#!/bin/sh
#python3 app.py
docker run -it --name pytorch -v app:/app  -v models:/app/models -v images:/app/images mlimage sh 