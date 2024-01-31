from fastapi import FastAPI
from ml_logic.data_extraction import download_wine_file
from fastapi.responses import FileResponse

app = FastAPI()


#http://127.0.0.1:8000/
@app.get("/")
def root():
    return {"Hello":"World"}

#http://127.0.0.1:8000/wine-raw
@app.get("/wine-raw")
def download_wine():
    download_wine_file('the_public_bucket','wine-clustering.csv','../data/wine_raw.csv')
    return FileResponse('../data/wine_raw.csv')
