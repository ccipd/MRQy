# !! EDIT THIS: Set base Docker image to be used.
FROM python:3.8.17-bullseye as wheelbuilder

COPY requirements.txt ./

RUN --mount=type=cache,target=/.cache/pip \
    python -m pip install -U pip setuptools wheel \
 && pip wheel --no-deps --cache-dir=/.cache/pip --wheel-dir /wheels -r requirements.txt


FROM python:3.8.17-slim-bullseye

COPY --from=wheelbuilder /wheels /wheels
RUN python -m venv /venv \
 && . /venv/bin/activate \
 && python -m pip install --no-cache -U pip setuptools wheel \
 && pip install --no-cache /wheels/*

ARG UID=5642
ARG GID=5642

RUN ( getent group $GID >/dev/null || groupadd -r -g $GID localgroup ) \
 && useradd -m -l -s /bin/bash -g $GID -N -u $UID localuser \
 && chown -R $UID:$GID /venv

WORKDIR /home/localuser
USER localuser

# !! EDIT THIS: Copy the needed local files (code and otherwise) into the container work directory.
# Note: For directories, have a separate COPY command for each top-level directory.
COPY --chown=$UID:$GID src RhinoHealth/run-mrqy.sh ./

# !! EDIT THIS: Ensure files are executable.
RUN chmod +x run-mrqy.sh

# This is basically what venv/bin/activate does
ENV PATH="/venv/bin:$PATH"
ENV VIRTUAL_ENV="/venv"
ENV PYTHONUNBUFFERED=1

# !! EDIT THIS: Set the command to be executed when the container is run, with arguments as needed.
CMD ["./run-mrqy.sh", "/input", "/output", "/input/cohort_data.csv"]
