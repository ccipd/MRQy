FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install -r requirements.txt
COPY . .
ENV PYTHONPATH "${PYTHONPATH}:/app/src"
