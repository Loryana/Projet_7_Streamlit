import streamlit as st
from PIL import Image
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import re
import joblib
import shap
import requests
from io import StringIO, BytesIO

import streamlit.components.v1 as components


API_URL = "http://13.61.10.141:8080//predict/"
#API_URL = "http://127.0.0.1:8000//predict/"

st.set_page_config("Dashboard de credit scoring - Projet 7 Data Scientist OpenClassrooms", layout = "wide")

number_of_ind = 1000

@st.cache_data(persist=True)
def load_data():
    df = pd.read_csv('df.csv', index_col = 0, nrows=number_of_ind)

    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    X = df.drop(columns = ['TARGET'])
    y = df[['TARGET']]

    explainer = joblib.load('explainer.pkl')

    return df, X, y, explainer
    
df, X, y, explainer = load_data()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


if "feat_selected" not in st.session_state:
    st.session_state.feat_selected = X.columns[0]

with st.sidebar:
    
    img = Image.open('logo.PNG')
    st.image(img)

    st.title("Société 'Prêt à dépenser'")
    st.subheader('Réponse à la demande de crédit')
    
    # Sélection du client
    id_filter = st.number_input("Sélectionnez l'identifiant de la personne :",value = 0 )

    st.divider()
    
    if "pred_clicked" not in st.session_state:
        st.session_state.pred_clicked = False

    if st.button("Prédire"):
        st.session_state.pred_clicked = True

if st.session_state.pred_clicked:
    ligne = X.iloc[id_filter]
    input_data = ligne.to_dict()

    response = requests.post(API_URL, json=input_data)

    prediction = response.json().get("predictions", "Erreur dans la réponse")
    pred = (prediction[0] >= 0.38)
    
    seuil = str(((0.38)*100))
    pourcentage = round(((prediction[0])*100),2)

    tab1, tab2, tab3, tab4 = st.tabs(["Résultats", "Informations clients", "Features", "Analyse bivariée"])
    with tab1 :
        if pred:
            st.balloons()
            
            a, b = st.columns(2)
            #a.metric(label = " Score du client :", value = pourcentage, delta = seuil, border = True)
            #Jauge
            with a :
                st.success(f"Résultat de la prédiction : Prêt accordé")
            with b:
                fig = go.Figure(go.Indicator(domain={'row': 0, 'column': 0},
                value=pourcentage,
                mode="gauge+number+delta",
                title={'text': "Visualisation du score du client", 'font_color':'black'},
                delta={'reference': float(seuil), "increasing": {"color": "green"}, 'decreasing': {"color": "red"}},
                gauge={'axis': {'range': [None, 100]},
                'steps': [{'range': [0, 100], 'color': "lightgray"}],
                'bar': {'color': "black"},
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': float(seuil)}}))
                
                fig.update_layout(paper_bgcolor="white", height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.snow()
            #st.success(f"Résultat de la prédiction : Prêt refusé")
            a, b = st.columns(2)
            with a : 
                st.write(f"Résultat de la prédiction : Prêt refusé")
            with b :
                #Jauge
                fig = go.Figure(go.Indicator(domain={'row': 0, 'column': 0},
                value=pourcentage,
                mode="gauge+number+delta",
                title={'text': "Visualisation du score du client", 'font_color':'black'},
                delta={'reference': float(seuil), "increasing": {"color": "green"}, 'decreasing': {"color": "red"}},
                gauge={'axis': {'range': [None, 100]},
                'steps': [{'range': [0, 100], 'color': "lightgray"}],
                'bar': {'color': "black"},
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': float(seuil)}}))
                
                fig.update_layout(paper_bgcolor="white", height=350)
                st.plotly_chart(fig, use_container_width=True)
                #st.metric(label = " Score du client :", value = pourcentage, delta = seuil, delta_color = "inverse", border = True)
    with tab2 :
            a_bar, b_bar = st.columns(2)
            with a_bar:
                st.write(" Informations du client numéro : " + str(id_filter))
                st.dataframe(X.iloc[id_filter, :])

            with b_bar:
                list_feature_names = X.columns.to_list()
                feat_select_plot_bar = st.selectbox(
                            "Sélectionnez une variable :", 
                            list_feature_names, 
                            )
                
                fig, ax = plt.subplots(figsize=(5, 5))
                sns.histplot(X[feat_select_plot_bar], bins=30, kde=True, color='skyblue', edgecolor='black', ax=ax)
                ax.axvline(X[feat_select_plot_bar].iloc[id_filter], color='red', linestyle='dashed', linewidth=2, label=f'Valeur pour le client {id_filter}')
                ax.set_title(f"Distribution de la variable {feat_select_plot_bar}")
                ax.set_xlabel("")
                ax.set_ylabel("Fréquence")
                ax.legend(loc='upper right')
                st.pyplot(fig)
    
    with tab3 :
        
        list_feature_names = X.columns.to_list()

        feat_select_plot1 = st.selectbox(
                    "Sélectionnez une feature pour x :", 
                    list_feature_names, 
                    )
        feat_select_plot2 = st.selectbox(
                    "Sélectionnez une feature pour y :", 
                    list_feature_names, 
                    )

        fig = go.Figure(data=go.Scatter(
            x=X[feat_select_plot1],
            y=X[feat_select_plot2],
            mode='markers',
            name="",
        ))

        fig.add_trace(go.Scatter(
            x=[X[feat_select_plot1].iloc[id_filter]],
            y=[X[feat_select_plot2].iloc[id_filter]],
            mode="markers",
            marker_symbol = 'diamond',
            marker=dict(color="salmon", size=12),
            name="Client",
            ))
        fig.update_layout(
            title="Scatter Plot",
            xaxis_title=feat_select_plot1,
            yaxis_title=feat_select_plot2,
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)


    with tab4 :
        
        shap_values = explainer(X)
        #st_shap(shap.summary_plot(shap_values, X, plot_size=[15,8]))

        force_plot_html = shap.force_plot(
            base_value=explainer.expected_value, 
            shap_values=shap_values.values[id_filter, :], 
            features=X.iloc[id_filter, :],
            matplotlib=False,
            )

        html_code = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
        components.html(html_code, height=400)
        #st.dataframe(X.iloc[id_filter, :])
        #st_shap(shap.force_plot(explainer.expected_value, shap_values.values[:number_of_ind, :], X.iloc[:number_of_ind, :]), height=400)

        list_feature_names = df.columns.to_list()

        a_shap, b_shap = st.columns(2)
        with a_shap:
        #shap.initjs()
        
            fig, ax = plt.subplots(figsize=(10, 10))
            shap.summary_plot(shap_values, features=X, feature_names=X.columns)

            st.pyplot(fig)
        
        

               

