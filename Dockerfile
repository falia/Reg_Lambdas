FROM public.ecr.aws/lambda/python:3.11

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY app.py .
COPY embedding_provider/ ./embedding_provider/
COPY milvus_provider/ ./milvus_provider/

# Set the Lambda handler
CMD ["app.lambda_handler"]