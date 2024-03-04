PORT ?= 8000
CONTAINER := knowledge_graph_validator
VERSION ?= `cat VERSION`
NUM_WORKERS ?= 1
DEMO_NAME ?= new-demo
LOG_LEVEL ?= DEBUG


.PHONY: dev
dev:
	pip install -e .
	pip install -r requirements.txt


.PHONY: demo
demo:
	@python -m quickstarter.create_demo --name $(DEMO_NAME)
	make run --directory demos/$(DEMO_NAME)


.PHONY: test
test:
	pytest knowledge_graph_validator
	flake8 knowledge_graph_validator --exclude *_pb2.py


# optional; if protobuf schema is used
.PHONY: proto
proto:
	protoc --python_out=knowledge_graph_validator schema.proto


# run service locally
.PHONY: run
run:
	gunicorn -b:$(PORT) 'knowledge_graph_validator.serving:run_app()' --log-level=$(LOG_LEVEL)


# service must be running, i.e. start "make run" first
.PHONY: example-request-reverse
example-request-reverse:
	curl -i -H "Content-Type: application/json" -X POST --data "@./examples/example_request.json" "http://localhost:8000/reverse"


.PHONY: example-request-count
example-request-count:
	curl -i -H "Content-Type: application/json" -X POST --data {} "http://localhost:8000/count"


# build docker container
.PHONY: build
build:
	docker build --no-cache -t $(CONTAINER):$(VERSION) -f Dockerfile .
	@echo "To run the container, use:"
	@echo "docker run -p 8000:8000 -e --rm -it $(CONTAINER):$(VERSION)"


# Initialise version control
.PHONY: initialise-version-control 
initialise-version-control:
	git init
	dvc init


# Add DVC remote storage
.PHONY: add-dvc-remote
add-dvc-remote:
	dvc remote add ${NAME} ${PATH}
