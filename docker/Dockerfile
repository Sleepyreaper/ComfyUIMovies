FROM python:3.12-slim

# ffmpeg is needed only for the optional ElevenLabs music merge.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY comfymovies ./comfymovies
COPY workflows ./workflows

RUN pip install --no-cache-dir .

# Defaults target the PC ComfyUI; override at run time.
ENV COMFY_HOST=192.168.1.90 \
    COMFY_PORT=8188 \
    COMFYMOVIES_OUTPUT=/app/output

VOLUME ["/app/output"]
ENTRYPOINT ["comfymovies"]
CMD ["--help"]
