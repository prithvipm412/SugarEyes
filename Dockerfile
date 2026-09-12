# FastAPI serving container for the SugarEyes screening API.
# CPU-only (see AGENTS.md: never assume CUDA) -- inference runs at the same
# ~1s/image observed locally on Apple M4 MPS, since the underlying PyTorch
# ops are the same; only the device backend differs.
FROM python:3.11-slim

WORKDIR /app

# Playwright's headless Chromium needs these system libraries for the PDF
# report renderer (see src/drscreen/report/build.py -- WeasyPrint's
# pango/cairo deps don't load cleanly even on macOS dev, so this project
# uses Playwright everywhere, including here).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

COPY src/ src/
COPY api/ api/
COPY configs/ configs/
COPY models/manifest.json models/manifest.json
# Trained checkpoints (models/*/best.pt) are not baked into the image --
# see docker-compose or the deploy docs for how they get mounted/fetched.
# This keeps the image buildable without >1GB of binary weights checked
# into (or pulled during) the image build itself.

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
