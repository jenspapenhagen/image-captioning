#!/bin/sh
gunicorn --workers=2 api:app