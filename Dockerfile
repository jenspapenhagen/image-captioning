FROM bitnami/pytorch:1.13.1-debian-11-r25

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./app.py ./

RUN mkdir -p /app/images
VOLUME /images

RUN mkdir -p /app/models/transformers
VOLUME /models

EXPOSE 5000
CMD ["python3", "./app/app.py"]
