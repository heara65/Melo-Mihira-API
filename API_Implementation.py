import fastapi
import uvicorn
import ssl
import tensorflow
from fastapi import *
import os

import ModelLoading
import uuid

IMAGEDIR = "Images/"

app = FastAPI()


def GenderMapper(gender_):
    genders = ['m', 'f']
    if gender_ == 'female':
        return genders[1]
    else:
        return genders[0]


@app.get('/')
def hello_world():
    return "Hello World" # to test


@app.post("/demographicsImage")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  

    # example of how you can save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    filepath = IMAGEDIR + file.filename
    # filepath = IMAGEDIR + "hell.jpg"
    Result = ModelLoading.finalImageOutput(filepath)
    # Result = ModelLoading.finalImageOutput(contents)
   

    return {"Gender": Result[1], "Age": Result[0], "Ethnicity": Result[2]
            
            }


