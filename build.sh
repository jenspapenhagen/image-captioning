#!/bin/sh
echo Building the Docker Image from Dockerfile
cat Dockerfile
echo lets go
docker build -t mlimage .