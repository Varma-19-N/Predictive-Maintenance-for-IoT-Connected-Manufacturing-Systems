# 🐳 STEP 1: Start from a clean Python image
FROM python:3.10-slim

# 🐳 STEP 2: Set working directory inside container
WORKDIR /app

# 🐳 STEP 3: Copy requirements first (for caching)
COPY requirements.txt .

# 🐳 STEP 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 🐳 STEP 5: Copy your entire project into container
COPY . .

# 🐳 STEP 6: Expose port 8501 (Streamlit’s default port)
EXPOSE 8501

# 🐳 STEP 7: Run the Streamlit app when container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]