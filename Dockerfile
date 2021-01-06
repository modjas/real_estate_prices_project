FROM continuumio/miniconda3:4.6.14
WORKDIR /app
COPY main.py ./main.py
COPY lgbm_pipeline.pkl ./lgbm_pipeline.pkl
#COPY environment.yml ./environment.yml
#RUN conda env create -f environment.yml

COPY requirements_conda.txt ./requirements_conda.txt
COPY heroku.yml ./heroku.yml

RUN while read requirement; do conda install --yes $requirement; done < requirements_conda.txt
RUN pip install streamlit
RUN conda install -c conda-forge lightgbm

COPY startup.sh startup.sh

#ENTRYPOINT "./startup.sh"
CMD ["streamlit", "run", "main.py"]
