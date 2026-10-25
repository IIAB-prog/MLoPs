from fastapi.testclient import TestClient
from api import app
client=TestClient(app)
def sample(): return {"age_ans":15,"profondeur_m":90,"debit_m3h":4.2,"usagers_jour":450,"jours_depuis_maintenance":300,"pannes_12m":1,"type_pompe":"solaire","region":"Centre"}
def test_health(): assert client.get("/health").status_code==200
def test_prediction_contract():
 r=client.post("/predict",json=sample())
 assert r.status_code==200, r.text
 d=r.json(); assert 0<=d["risque_panne_90j"]<=1; assert d["classe_risque"] in ["faible","modere","eleve"]
def test_invalid_input_rejected():
 x=sample(); x["age_ans"]=-1
 assert client.post("/predict",json=x).status_code==422
