from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd



# load the model
model=joblib.load('./Model/best_diabetes_model.pkl')


app=FastAPI(title='Diabetes Prediction API')

# input shcema
class DiabetesInput(BaseModel):
    HighBP: float
    HighChol: float
    CholCheck: float
    BMI: float
    Smoker: float
    Stroke: float
    HeartDiseaseorAttack: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    GenHlth: float
    MentHlth: float
    PhysHlth: float
    DiffWalk: float
    Sex: float
    Age: float
    Education: float
    Income: float


# process the input data
def process_data(data:dict):
    df=pd.DataFrame([data])

    # drop unnecessary columns
    cols_del = ["Sex", "Smoker", "Fruits", "Veggies", "NoDocbcCost", "AnyHealthcare", "CholCheck", "Education"]
    df = df.drop(cols_del, axis=1)

    return df



@app.get('/')
def root():
    return {'message':'Welcome to the Diabetes Prediction API'}

@app.post('/predict')
def predict(data:DiabetesInput):

    input_df=process_data(data.dict())

    # make prediction
    prediction=model.predict(input_df)
    prediction_proba=model.predict_proba(input_df)


    return{
        'prediction': int(prediction[0]),
        'probability': {
            'diabetes': float(prediction_proba[0][1]),
            'no diabetes': float(prediction_proba[0][0])
        }
    }



