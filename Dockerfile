FROM continuumio/miniconda3:4.6.14
RUN adduser -D appuser
USER appuser
WORKDIR /app

COPY --chown=appuser:appuser main.py ./main.py
COPY --chown=appuser:appuser lgbm_pipeline.pkl ./lgbm_pipeline.pkl
#COPY environment.yml ./environment.yml
#RUN conda env create -f environment.yml

COPY --chown=appuser:appuser requirements_conda.txt ./requirements_conda.txt
COPY --chown=appuser:appuser heroku.yml ./heroku.yml

RUN while read requirement; do conda install --yes $requirement; done < requirements_conda.txt
RUN pip install streamlit
RUN conda install -c conda-forge lightgbm

COPY --chown=appuser:appuser startup.sh startup.sh

#ENTRYPOINT "./startup.sh"
CMD ["streamlit", "run", "main.py"]
