from fastapi import FastAPI
import uvicorn
import pickle
from pydantic import BaseModel

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

app = FastAPI()
model = pickle.load(open('clf.pkl','rb'))

@app.get('/')
def home():
    return {"hello":"stranger"}

@app.post('/predict')
def predict_notes(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = model.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1",port=8080)
