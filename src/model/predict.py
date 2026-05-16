import joblib
from xgboost import XGBClassifier
import streamlit as st

@st.cache_resource
def load_model_and_scaler(model_path='modelo/modelo.json', scaler_path='scaler/scaler.pkl'):
    """
    Carrega o modelo XGBoost e o scaler salvos no disco.
    Utiliza cache do Streamlit para evitar recarregar a cada interação.
    """
    model = XGBClassifier()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def make_prediction(model, scaler, input_df):
    """
    Escalona os dados de entrada e realiza a predição.
    Retorna a classe predita e a probabilidade da classe positiva.
    """
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob > 0.5 else 0
    return prediction, prob
