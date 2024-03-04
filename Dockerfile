FROM python:3.10-bullseye

WORKDIR /srv

RUN pip install --upgrade pip

ADD knowledge_graph_validator ./knowledge_graph_validator
ADD requirements.txt ./
ADD VERSION ./
ADD setup.py ./
ADD Makefile ./

RUN make dev

CMD make run
