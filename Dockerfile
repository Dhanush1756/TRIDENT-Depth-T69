FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
COPY requirements.txt requirements-app.txt requirements-learning-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt
ARG INSTALL_LEARNING=0
RUN if [ "$INSTALL_LEARNING" = "1" ]; then pip install --no-cache-dir -r requirements-learning-cpu.txt; fi
COPY trident_depth ./trident_depth
COPY backend ./backend
COPY frontend ./frontend
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/health', timeout=4)"
CMD ["python", "-m", "backend.serve"]
