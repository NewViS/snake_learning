
FROM python:3.10-bullseye
LABEL maintainer="qweewq258456@gmail.com"


WORKDIR /usr/app/front
EXPOSE 3000


COPY ./ ./


RUN python:3.10 install


CMD ["python3", "./snake.py"]
