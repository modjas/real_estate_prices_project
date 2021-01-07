FROM continuumio/miniconda3:4.8.2

WORKDIR /app
EXPOSE 8501

COPY main.py ./main.py
COPY lgbm_pipeline.pkl ./lgbm_pipeline.pkl

RUN conda install pandas=1.0.5 joblib=0.16.0
RUN pip install streamlit
RUN conda install -c conda-forge lightgbm

CMD ["streamlit", "run", "main.py"]