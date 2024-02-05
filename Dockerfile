####### 👇 OPTIMIZED SOLUTION (x86)👇 #######


# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
#FROM tensorflow/tensorflow
#:2.10.0
FROM python:3.10.6-buster
# OR for apple silicon, use this base image instead
# FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

# WORKDIR /prod
WORKDIR /root/code/juancruzgui/Projects/wine-analysis-romboost

#Install the google SDK
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud
RUN tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz
RUN /usr/local/gcloud/google-cloud-sdk/install.sh
RUN /usr/local/gcloud/google-cloud-sdk/bin/gcloud init
RUN gcloud auth activate-service-account --key-file = wine-package/Credentials/google_credentials.json


COPY .env .env
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN gcloud auth application-default login

COPY wine-package wine-package
COPY setup.py setup.py
RUN pip install .


CMD uvicorn wine-package.app.api:app --host 127.0.0.1 --port 8000 && python wine-package/app/main.py
