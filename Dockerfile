FROM python:3.9.12

USER root

RUN apt-get update \
    && pip install --upgrade pip


WORKDIR /App

COPY . /App

RUN pip3 install --upgrade -r PredMntec_CbV_AI/requirements.txt

RUN python3 PredMntec_CbV_AI/setup.py install

EXPOSE 8000

RUN python3 -m PredMntec_CbV_Restapp.launch

ENTRYPOINT [ "python3","-m PredMntec_CbV_Restapp.launch" ]





