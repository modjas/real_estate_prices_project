FROM continuumio/miniconda3:4.8.2
WORKDIR /app
RUN chmod -R 777 /opt/conda
RUN chmod -R 777 /app

COPY main.py ./main.py
COPY lgbm_pipeline.pkl ./lgbm_pipeline.pkl
#COPY environment.yml ./environment.yml
#RUN conda env create -f environment.yml

#COPY --chown=appuser:appuser requirements_conda.txt ./requirements_conda.txt
COPY heroku.yml ./heroku.yml

#RUN while read requirement; do conda install --yes $requirement; done < requirements_conda.txt
RUN pip install streamlit
RUN conda install -c conda-forge lightgbm
RUN conda install -y pandas=1.0.5 bs4=4.9.1

COPY startup.sh startup.sh

RUN adduser appuser
USER appuser

#ENTRYPOINT "./startup.sh"
CMD ["streamlit", "run", "main.py"]
