import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# configuração da página
st.set_page_config(page_title="Preditor de Compras", page_icon="💰")

# função para carregar os arquivos uma única vez
@st.cache_resource
def load_model_and_scaler():
    model = XGBClassifier()
    model.load_model('modelo/modelo.json')
    scaler = joblib.load('scaler/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model_and_scaler()

    st.title("🛒 Inteligência de Vendas: Previsão de Compra")
    st.markdown("Preveja se um visitante do site irá concluir uma transação com base no comportamento de navegação.")

    with st.form("user_data_form"):
        st.subheader("Comportamento do Usuário")
        
        col1, col2 = st.columns(2)
        
        with col1:
            page_values = st.number_input("Valor da Página (PageValues)", min_value=0.0, help="Valor médio da página visitada.")
            total_duration = st.number_input("Duração Total (Segundos)", min_value=0.0, help="Tempo total gasto no site.")
            month = st.selectbox("Mês", options=list(range(2, 13)), format_func=lambda x: ['Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'][x-2])

        with col2:
            exit_rates = st.slider("Taxa de Rejeição (ExitRates)", 0.0, 1.0, 0.05)
            visitor_type = st.radio("Tipo de Visitante", ["Novo", "Recorrente", "Outro"])
            
        submit = st.form_submit_button("Analisar Probabilidade de Compra")

    if submit:
        # mapeando inputs para o formato do modelo
        is_returning = 1 if visitor_type == "Recorrente" else 0
        is_other = 1 if visitor_type == "Outro" else 0
        
        # criando o DataFrame na ordem correta das colunas
        # ordem: ['PageValues', 'ExitRates', 'Total_Duration', 'Month', 'VisitorType_Returning_Visitor', 'VisitorType_Other']
        features = ['PageValues', 'ExitRates', 'Total_Duration', 'Month', 'VisitorType_Returning_Visitor', 'VisitorType_Other']
        input_df = pd.DataFrame([[page_values, exit_rates, total_duration, month, is_returning, is_other]], columns=features)
        
        # escalonamento e Predição
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = 1 if prob > 0.5 else 0

        st.divider()
        if prediction == 1:
            st.balloons()
            st.success(f"### ✅ Alta Chance de Conversão!\nProbabilidade de compra: **{prob:.2%}**")
            st.info("Sugestão: Oferecer um cupom de frete grátis para garantir o fechamento!")
        else:
            st.warning(f"### ❌ Baixa Probabilidade de Compra\nProbabilidade: **{prob:.2%}**")
            st.info("Sugestão: Remarketing focado em benefícios do produto.")

except Exception as e:
    st.error(f"Erro ao carregar o modelo ou processar dados: {e}")
