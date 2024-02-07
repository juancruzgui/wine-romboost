####### 👇 OPTIMIZED SOLUTION (x86)👇 #######


# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
#FROM tensorflow/tensorflow
#:2.10.0
FROM python:3.10.6-buster
# OR for apple silicon, use this base image instead
# FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

# WORKDIR /prod
WORKDIR /root/code/juancruzgui/Projects/wine-analysis-romboost

#Copy
COPY .env .env
COPY requirements.txt requirements.txt
COPY wine-package wine-package
COPY setup.py setup.py

# #GCP AUTH
# #Install the google SDK
# RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
# RUN mkdir -p /usr/local/gcloud
# RUN tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz
# RUN /usr/local/gcloud/google-cloud-sdk/install.sh
# # Set the PATH to include gcloud binary
# ENV PATH=/usr/local/gcloud/google-cloud-sdk/bin:$PATH
# #RUN gcloud auth activate-service-account --key-file = wine-package/Credentials/google_credentials.json
# # Assuming the service account key file is in the wine-package/Credentials directory
# ENV GOOGLE_APPLICATION_CREDENTIALS=wine-package/Credentials/google_credentials.json
# # Set project and zone non-interactively
# RUN gcloud config set project simple-sales-412318
# #RUN gcloud config set compute/zone YOUR_COMPUTE_ZONE
# # Activate service account non-interactively
# RUN gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS



RUN pip install --upgrade pip
RUN pip install -r requirements.txt
#RUN gcloud auth application-default login

RUN pip install .


CMD ["uvicorn", "wine-package.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
