FROM python:3.11-slim
WORKDIR /app
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
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
