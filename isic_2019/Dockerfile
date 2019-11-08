FROM ubuntu:18.04

MAINTAINER Jenny Yu "jypucca@hotmail.com"

RUN apt-get update -y && \
    apt-get install -y python3 python3-dev python3-pip

COPY . /app

WORKDIR /app

RUN mkdir img_storage

RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

EXPOSE 5000

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]
