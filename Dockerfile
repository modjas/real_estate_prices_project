FROM continuumio/miniconda3
WORKDIR /app
COPY main.py ./main.py
COPY test.pkl ./test.pkl
#COPY environment.yml ./environment.yml
#RUN conda env create -f environment.yml

COPY requirements_conda.txt ./requirements_conda.txt

RUN while read requirement; do conda install --yes $requirement; done < requirements_conda.txt
RUN pip install streamlit
RUN conda install -c conda-forge lightgbm

COPY startup.sh startup.sh

ENTRYPOINT "./startup.sh"