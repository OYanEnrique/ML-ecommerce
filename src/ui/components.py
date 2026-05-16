import streamlit as st
import pandas as pd

def render_header():
    """Renderiza o cabeçalho principal da aplicação."""
    st.title("🛒 Inteligência de Vendas: Previsão de Compra")
    st.markdown("Preveja se um visitante do site irá concluir uma transação com base no comportamento de navegação.")

def render_form():
    """
    Renderiza o formulário de entrada de dados do usuário.
    Retorna um DataFrame Pandas se o formulário for submetido, caso contrário retorna None.
    """
    with st.form("user_data_form"):
        st.subheader("Comportamento do Usuário")
        
        col1, col2 = st.columns(2)
        
        with col1:
            page_values = st.number_input(
                "Valor da Página (PageValues)", 
                min_value=0.0, 
                help="Métrica que representa o valor médio das páginas web que o usuário visitou. Valores maiores indicam navegação em páginas mais valiosas/produtos mais caros."
            )
            total_duration = st.number_input(
                "Duração Total (Segundos)", 
                min_value=0.0, 
                help="Tempo total, em segundos, que o usuário passou navegando e interagindo com o site."
            )
            month = st.selectbox(
                "Mês da Visita", 
                options=list(range(2, 13)), 
                format_func=lambda x: ['Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'][x-2],
                help="Mês em que a sessão ocorreu. A sazonalidade (como feriados ou datas comemorativas) pode influenciar muito as chances de compra."
            )

        with col2:
            exit_rates = st.slider(
                "Taxa de Rejeição (ExitRates)", 
                0.0, 1.0, 0.05,
                help="A porcentagem de visualizações de página onde a página foi a última da sessão. Valores altos indicam que as pessoas abandonam o site frequentemente."
            )
            visitor_type = st.radio(
                "Tipo de Visitante", 
                ["Novo", "Recorrente", "Outro"],
                help="Indica o histórico de visitas do usuário. Clientes 'Recorrentes' podem ter maior familiaridade e intenção de compra do que visitantes 'Novos'."
            )
            
        submit = st.form_submit_button("Analisar Probabilidade de Compra")

    if submit:
        # Mapeando inputs para o formato do modelo
        is_returning = 1 if visitor_type == "Recorrente" else 0
        is_other = 1 if visitor_type == "Outro" else 0
        
        # Criando o DataFrame na ordem correta das colunas exigida pelo modelo
        features = ['PageValues', 'ExitRates', 'Total_Duration', 'Month', 'VisitorType_Returning_Visitor', 'VisitorType_Other']
        input_df = pd.DataFrame([[page_values, exit_rates, total_duration, month, is_returning, is_other]], columns=features)
        
        return input_df
    
    return None

def render_result(prediction, prob):
    """Renderiza o resultado da predição na interface."""
    st.divider()
    if prediction == 1:
        st.balloons()
        st.success(f"### ✅ Alta Chance de Conversão!\nProbabilidade de compra: **{prob:.2%}**")
        st.info("Sugestão: Oferecer um cupom de frete grátis para garantir o fechamento!")
    else:
        st.warning(f"### ❌ Baixa Probabilidade de Compra\nProbabilidade: **{prob:.2%}**")
        st.info("Sugestão: Remarketing focado em benefícios do produto.")
