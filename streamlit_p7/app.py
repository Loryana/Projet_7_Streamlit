import streamlit as st
from PIL import Image
import pandas as pd 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import re
import shap
import requests
import boto3
from io import StringIO

API_URL = "http://0.0.0.0:8000/predict/"

st.set_page_config("Projet 7 OC", layout = "wide")

@st.cache_data(persist=True)
def load_data():
    df = pd.read_csv('df.csv', index_col = 0)

    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    X = df.drop(columns = ['TARGET'])
    y = df[['TARGET']]

    return df, X, y
    
df, X, y = load_data()

def run_main():
    with st.sidebar:
        
        img = Image.open('logo.PNG')
        st.image(img)

        st.title("Société 'Prêt à dépenser'")
        st.subheader('Réponse à la demande de crédit')
        
        # Sélection du client
        id_filter = st.number_input("Sélectionnez l'identifiant de la personne :",value = 0 )

    if st.button("Prédire"):
        ligne = X.iloc[id_filter]
        input_data = ligne.to_dict()

        response = requests.post(API_URL, json=input_data)

        prediction = response.json().get("predictions", "Erreur dans la réponse")
        #st.success(f"Résultat de la prédiction  {int(prediction[0])}")

        if int(prediction[0]) == 0:
            st.snow()
            st.success(f"Résultat de la prédiction : Prêt refusé")
        else:
            st.balloons()
            st.success(f"Résultat de la prédiction : Prêt accordé")
        
    


run_main()