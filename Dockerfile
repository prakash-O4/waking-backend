FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy entire project
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Ensure correct Python path
ENV PYTHONPATH=.

# Expose port 8000
EXPOSE 8000

# Change from Lambda handler to uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]