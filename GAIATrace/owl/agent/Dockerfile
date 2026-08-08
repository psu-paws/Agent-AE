# Playwright base: the browser toolkit needs a working chromium + deps.
FROM mcr.microsoft.com/playwright/python:v1.57.0-jammy
USER root

WORKDIR /app

# requirements.txt alone first, so editing code doesn't invalidate the pip layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chown -R pwuser:pwuser /app

USER pwuser
ENTRYPOINT ["python", "run_gaia_workforce_vllm_planner.py"]
