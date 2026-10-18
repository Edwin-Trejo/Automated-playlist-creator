FROM python:3.11-slim

# ffmpeg decodes iTunes .m4a previews; libgomp1 is needed by LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY data/dataset.csv data/genre_groups.json data/

# The song cache lives in a volume so rebuilds keep it.
RUN useradd --system --uid 1000 app && mkdir /data && chown app /data
USER app
ENV PYTHONUNBUFFERED=1 DB_PATH=/data/app.db
VOLUME /data

CMD ["python", "-m", "src.web.app"]
