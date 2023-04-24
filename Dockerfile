# Set base image
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy poetry files to container
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-root --no-dev

# Copy app files to container
COPY streamlit_app.py ./
COPY 32419-cyanotype-curve.acv ./

# Install required packages
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6

# Expose port
EXPOSE 8501

# Start app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
