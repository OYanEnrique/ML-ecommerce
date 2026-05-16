import streamlit as st
from src.model.predict import load_model_and_scaler, make_prediction
from src.ui.components import render_header, render_form, render_result

# Configuração da página (deve ser a primeira chamada do Streamlit)
st.set_page_config(page_title="Preditor de Compras", page_icon="💰")

def main():
    try:
        # Carregando o modelo e scaler (com cache)
        model, scaler = load_model_and_scaler()
        
        # Renderizando a interface
        render_header()
        input_df = render_form()
        
        # Se o formulário foi preenchido e submetido
        if input_df is not None:
            prediction, prob = make_prediction(model, scaler, input_df)
            render_result(prediction, prob)

    except Exception as e:
        st.error(f"Erro ao carregar o modelo ou processar dados: {e}")

if __name__ == "__main__":
    main()
