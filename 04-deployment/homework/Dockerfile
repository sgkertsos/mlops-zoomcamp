FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .

RUN pip install pyarrow numpy==1.21.5 pandas==1.4.2 scikit-learn==1.0.2

RUN chmod +x starter.py


ENTRYPOINT [ "python", "starter.py" ]