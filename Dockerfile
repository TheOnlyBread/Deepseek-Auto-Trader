# Use official lightweight Python
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (Render overrides this in web vs worker)
CMD ["python", "deepseek_binance_autotrader.py"]
