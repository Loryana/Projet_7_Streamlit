import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go

id_filter = 0

df = pd.read_csv('df.csv', index_col = 0, nrows=1000)
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

X = df.drop(columns = ['TARGET'])
y = df[['TARGET']]

st.write("### Test de la liste déroulante")

# Initialiser session_state pour éviter que l'affichage disparaisse
if "pred_clicked" not in st.session_state:
    st.session_state.pred_clicked = False

# Bouton pour déclencher l'affichage
if st.button("Prédire"):
    st.session_state.pred_clicked = True  # Stocker l'état du bouton

if st.session_state.pred_clicked:
    list_feature_names = X.columns.to_list()

    tab1, tab2, tab3, tab4 = st.tabs(["Résultats", "Informations clients", "Features", "Analyse bivariée"])

    with tab2:
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
        ))

        fig.add_trace(go.Scatter(
            x=[X[feat_select_plot1].iloc[id_filter]],
            y=[X[feat_select_plot2].iloc[id_filter]],
            mode="markers",
            marker_symbol = 'diamond',
            name="Client",
            ))
        fig.update_layout(
            title="Scatter Plot",
            xaxis_title=feat_select_plot1,
            yaxis_title=feat_select_plot2,
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)