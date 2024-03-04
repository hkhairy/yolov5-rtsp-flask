######################################## Stage 1 ######################################
FROM python:3.11 AS Builder

RUN pip install "poetry==1.7.1" && poetry self add "poetry-plugin-export==1.6.0"
ARG yolo_model="yolov5s.onnx"
RUN wget "https://github.com/ultralytics/yolov5/releases/download/v7.0/${yolo_model}" --output-document "yolov5.onnx"
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --without dev

COPY src src
COPY main.py main.py
RUN poetry install --only-root

RUN poetry export -f requirements.txt --output requirements.txt
RUN poetry build --format wheel

############################################## Stage 2 ########################################################
FROM python:3.11-slim AS Runner
RUN apt update && apt install && apt install -y ffmpeg 

COPY --from=Builder /yolov5.onnx /yolov5.onnx
COPY --from=Builder /requirements.txt /requirements.txt
COPY --from=Builder /dist /dist
COPY --from=Builder /main.py /main.py

RUN pip install -r /requirements.txt --no-cache-dir --no-deps
RUN pip install /dist/*.whl --no-cache-dir --no-deps

EXPOSE 8080
ENTRYPOINT [ "python", "main.py" ]
