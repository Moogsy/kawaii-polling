FROM python:3.10-slim

# Don’t buffer stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy in and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# If you haven’t already listed gunicorn in requirements.txt, install it now
RUN pip install --no-cache-dir gunicorn

# Bundle your application code
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Launch the app with Gunicorn (2 workers is a good starting point)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2"]

