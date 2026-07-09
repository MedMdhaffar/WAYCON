# Goal 6 stub. The service is currently run directly during development:
#   python -m forensics.face_engine.app
#
# Containerization will pin CUDA/runtime dependencies in Goal 6.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

EXPOSE 5010
CMD ["python", "-m", "forensics.face_engine.app"]

