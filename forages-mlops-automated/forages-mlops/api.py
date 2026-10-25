from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from config import MODEL_DIR, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
app=FastAPI(title="Forage Risk API",version="1.0.0",description="Estimation du risque de panne d'un forage dans les 90 jours.")
MODEL_PATH=MODEL_DIR/"forage_risk_model.joblib"
MODEL=joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
class ForageInput(BaseModel):
 age_ans:int=Field(ge=0,le=100); profondeur_m:float=Field(gt=0,le=1000); debit_m3h:float=Field(ge=0,le=500); usagers_jour:int=Field(ge=0,le=100000); jours_depuis_maintenance:int=Field(ge=0,le=5000); pannes_12m:int=Field(ge=0,le=100); type_pompe:Literal["manuelle","thermique","solaire"]; region:Literal["Centre","Est","Nord","Sud"]
class Prediction(BaseModel):
 risque_panne_90j:float; classe_risque:Literal["faible","modere","eleve"]; seuil:float; modele:str
@app.get("/health")
def health(): return {"status":"ok" if MODEL is not None else "model_not_loaded","model_loaded":MODEL is not None}
@app.get("/metadata")
def metadata():
 p=MODEL_DIR/"metadata.json"
 if not p.exists(): raise HTTPException(503,"Modèle indisponible")
 return json.loads(p.read_text(encoding="utf-8"))
@app.post("/predict",response_model=Prediction)
def predict(payload:ForageInput):
 if MODEL is None: raise HTTPException(503,"Modèle absent; exécuter python -m src.modeling")
 row=pd.DataFrame([payload.model_dump()])[FEATURES]; risk=float(MODEL.predict_proba(row)[0,1]); cls="eleve" if risk>=0.70 else ("modere" if risk>=0.40 else "faible")
 meta=json.loads((MODEL_DIR/"metadata.json").read_text(encoding="utf-8"))
 return Prediction(risque_panne_90j=round(risk,4),classe_risque=cls,seuil=meta["threshold"],modele=meta["model_name"])
