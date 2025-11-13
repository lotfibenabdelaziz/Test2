# -----------------------------
# Step 1: Base Image
# -----------------------------
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# -----------------------------
# Step 2: Copy and Install Dependencies
# -----------------------------
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Step 3: Copy Application Files
# -----------------------------
COPY . .

# -----------------------------
# Step 4: Expose Flask Port
# -----------------------------
EXPOSE 5000

# -----------------------------
# Step 5: Run Flask App
# -----------------------------
CMD ["python", "app.py"]
