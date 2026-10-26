FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Default: offline check with the mock model. Use --llm anthropic with ANTHROPIC_API_KEY set for the real agent.
CMD ["python", "eval.py", "--llm", "mock"]
