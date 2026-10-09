from fastapi import FastAPI, File, UploadFile
import shutil

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Le serveur Birder est en ligne !"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # TODO: Appeler Birder ici avec le modèle mvit_v2_t_il-all.pt

    return {"species": "Eurasian teal (Simulation)"}