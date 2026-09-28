FROM python:3.12-slim-trixie

# Fonts used to redraw prices on photos/scans (Vazirmatn = Persian digits);
# Tesseract OCR (Persian + English) lets the bot read photos without Gemini.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core fonts-liberation fonts-crosextra-carlito fonts-crosextra-caladea \
        fonts-vazirmatn libglib2.0-0 tesseract-ocr tesseract-ocr-fas tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["python", "bot.py"]
