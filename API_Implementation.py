import fastapi
import uvicorn
import ssl
import tensorflow
from fastapi import *
import os

import ModelLoading
# import activity
# import emotionModel
# import music_recommendation
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
    return "Hello World"


@app.post("/demographicsImage")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!

    # example of how you can save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    filepath = IMAGEDIR + file.filename
    Result = ModelLoading.finalImageOutput(filepath)
    # Recommendations = music_recommendation.recommendDemographics(Result[0], GenderMapper(Result[1]), Result[2])

    return {"Gender": Result[1], "Age": Result[0], "Ethnicity": Result[2]
            # , "Recommendations": Recommendations
            }


# @app.post("/emotionImage")
# async def create_upload_file(file: UploadFile = File(...)):
#     file.filename = f"{uuid.uuid4()}.jpg"
#     contents = await file.read()  # <-- Important!

#     # example of how you can save the file
#     with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
#         f.write(contents)

#     filepath = IMAGEDIR + file.filename
#     Result = emotionModel.input_image_Emotion(filepath)
#     Recommendations = music_recommendation.recommendEmotion(Result)

#     return {"Emotion": Result, "Recommendations": Recommendations}


# @app.post("/activityVideo")
# async def create_upload_file(file: UploadFile = File(...)):
#     file.filename = f"{uuid.uuid4()}.avi"
#     contents = await file.read()  # <-- Important!

#     # example of how you can save the file
#     with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
#         f.write(contents)

#     filepath = IMAGEDIR + file.filename
#     Result = activity.ActivityMusicRecommendation(filepath)
#     Recommendations = music_recommendation.recommendActivity(Result)

#     return {"Activity": Result, "Recommendations": Recommendations}


# if __name__ == "__main__":
#     uvicorn.run(app, port=8081, host='192.168.8.101')
