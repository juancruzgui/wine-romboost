from fastapi import FastAPI
from .ml_logic.data_extraction import download_wine_file
from fastapi.responses import FileResponse
import os
import sys
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
script_path = os.path.abspath(__file__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#http://127.0.0.1:8000/
@app.get("/")
def root():
    return {"Hello":"World"}

#http://127.0.0.1:8000/wine-raw
@app.get("/wine-raw")
def download_wine():
    download_wine_file('the_public_bucket','wine-clustering.csv',os.path.join(os.path.dirname(script_path),'..','data','wine_raw.csv'))
    return FileResponse(os.path.join(os.path.dirname(script_path),'..','data','wine_raw.csv'))


#http://127.0.0.1:8000/analysis-images
@app.get("/analysis-images")
def get_images_labels():
    files = os.listdir(os.path.join(os.path.dirname(script_path),'.','images'))
    files_dict = {f'file{i}':
        {'name':file.split('/')[-1],
         'url':f'http://127.0.0.1:8000/images?name={file.split("/")[-1]}'}
                  for i,file in enumerate(files)}
    return files_dict

#http://127.0.0.1:8000/images&name=''
@app.get("/images")
def get_images_labels(name:str):
    file = os.path.join(os.path.dirname(script_path),'.','images',f'{name}')
    return FileResponse(file)
