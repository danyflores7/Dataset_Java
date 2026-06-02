from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from predictor import ClonePredictor
from typing import List
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ClonePredictor()


@app.get("/")
def root():
    return {
        "status": "running"
    }


@app.post("/predict")
async def predict(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".java"
    ) as f1:

        f1.write(await file1.read())
        path1 = f1.name

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".java"
    ) as f2:

        f2.write(await file2.read())
        path2 = f2.name

    result = predictor.predict_files(
        path1,
        path2
    )

    os.remove(path1)
    os.remove(path2)

    return result


@app.post("/predict-folder")
async def predict_folder(
    files: List[UploadFile] = File(...)
):

    temp_files = []

    try:

        for file in files:

            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".java"
            )

            tmp.write(await file.read())
            tmp.close()

            temp_files.append(
                (
                    file.filename,
                    tmp.name
                )
            )

        results = []

        for i in range(len(temp_files)):
            for j in range(i + 1, len(temp_files)):

                name1, path1 = temp_files[i]
                name2, path2 = temp_files[j]

                prediction = predictor.predict_files(
                    path1,
                    path2
                )

                results.append(
                    {
                        "file1": name1,
                        "file2": name2,
                        "prediction": prediction["prediction"],
                        "confidence": prediction["confidence"]
                    }
                )

        return results

    finally:

        for _, path in temp_files:

            if os.path.exists(path):
                os.remove(path)