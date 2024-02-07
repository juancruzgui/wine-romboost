# wine-romboost
### Introduction
Unsupervised learning on wine dataset. Detecting clusters of wines and describing them.
## Analysis
Analysis is located in two jupyter notebooks on the folder wine-package/notebooks

 ---
 
## Docker Setup
### Dockerfile
Prior building the image we need to work on one line in the docker file:
- WORKDIR </root/code/juancruzgui/Projects/wine-analysis-romboost->your_path>: You will need to change this path to the path of the parent directory to the wine-package folder.
### Docker build
From the command line and located where the DockerFile is you will need to run the following line to build the image:

**docker build -t <image_name> .**

### Once the image is built:
- Once the image is built the image is going to run the API on https://127.0.0.1:8000
- To excecute the script that will perform the analysis you can use the terminal on docker desktop once the image is running and excecute:

**python wine-package/app/main.py**

![image](https://github.com/juancruzgui/wine-romboost/assets/71938321/a61d6f0e-0230-4f47-bc71-696be9888096)

---
## API DOC
FastAPI to download the raw dataset from GCP and images from analysis.

### endpoints:
- [0.0.0.0:8000](http://0.0.0.0:8000/) (method:GET) -> "Hello World"
- [0.0.0.0:](http://0.0.0.0:8000/wine-raw.csv)http://127.0.0.1:8000/wine-raw.csv (method:GET) -> Download wine_raw csv file
- [http://0.0.0.0:8000/analysis-images](http://0.0.0.0:8000/analysis-images) (method:GET) -> Returns a json dict with all images names and urls from the analysis performed.
![image](https://github.com/juancruzgui/wine-romboost/assets/71938321/f8ca353b-9b5a-4f7e-812a-3b5e46306de0)

- http://0.0.0.0:8000/images?name= <name:str> (method:GET) ->Returns the image corresponding to name - you can check image names on the endpoint above.
![image](https://github.com/juancruzgui/wine-romboost/assets/71938321/9335c179-c937-41bb-8eb1-223598da86f6)


