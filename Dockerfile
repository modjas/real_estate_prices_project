FROM python:3.7.6-slim

WORKDIR /app
EXPOSE 8501

COPY main.py lgbm_pipeline.pkl ./

RUN pip install pandas==1.0.5 streamlit lightgbm scikit-learn==0.23.1 && \
    apt-get update && apt-get install libgomp1 && \
    useradd -m realestate && \
    chown -R realestate /app && \
    chmod -R u+x /app && \
    rm -rf /var/lib/apt/lists/*

USER realestate

CMD ["streamlit", "run", "main.py"]