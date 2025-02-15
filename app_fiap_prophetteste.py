import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta

# Configurar título do app
st.set_page_config(page_title="Previsão do Preço do Petróleo", page_icon="📊", layout="wide")
st.title("📊 Previsão do Preço do Petróleo")

# Criar um menu lateral fixo
st.sidebar.header("Forecast")
menu = st.sidebar.radio(" ", ["Previsão"])

if menu == "Previsão":
    # Carregar o arquivo Brent.xlsx diretamente
    brent_file = "Brent.xlsx"
    
    try:
        df = pd.read_excel(brent_file, decimal=",")
    except FileNotFoundError:
        st.error("Erro: Arquivo 'Brent.xlsx' não encontrado! Certifique-se de que o arquivo está no diretório correto.")
        st.stop()
    
    df.columns = ['Data', 'Close']
    df['Data'] = pd.to_datetime(df['Data'])
    df.set_index('Data', inplace=True)
    
    # Exibir última data real para depuração
    st.write("Última data real no dataset:", df.index.max())

    # Transformar os dados no formato do Prophet
    df_prophet = df.reset_index()
    df_prophet.columns = ['ds', 'y']  # Prophet exige essas colunas
    
    # Criar o modelo Prophet
    model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.1)
    
    # Treinar o modelo
    model_prophet.fit(df_prophet)

    # Escolha de data para previsão
    st.subheader("📅 Escolha a Data de Previsão")
    last_real_date = df.index.max()
    future_date = st.date_input("Selecione a data para prever:", value=last_real_date + timedelta(days=30))

    if pd.Timestamp(future_date) <= last_real_date:
        st.warning(f"A previsão só pode ser feita para datas após {last_real_date.date()}.")
    else:
        # Calcular quantos dias prever com base na data escolhida
        forecast_days = (pd.Timestamp(future_date) - last_real_date).days
        
        # Criar DataFrame futuro com base na data escolhida
        future_prophet = model_prophet.make_future_dataframe(periods=forecast_days, freq='D')

        # Gerar previsões
        forecast_prophet = model_prophet.predict(future_prophet)

        # Filtrar previsões até a data escolhida
        forecast_prophet = forecast_prophet[forecast_prophet['ds'] <= pd.Timestamp(future_date)]

        # Exibir gráfico da previsão
        st.subheader("📈 Gráfico de Previsão com Prophet")
        
        fig1, ax = plt.subplots(figsize=(10, 4))
        model_prophet.plot(forecast_prophet, ax=ax)

        # Capturar os elementos do gráfico automaticamente
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[2], handles[0], handles[1]], 
                ["Intervalo de Confiança", "Dados Reais", "Previsão"], 
                loc="upper left", fontsize=8)

        ax.set_xlabel("Data", fontsize=8)
        ax.set_ylabel("Valor", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)
        
        st.pyplot(fig1)

        # Exibir tabela filtrada de previsões
        st.subheader("📊 Tabela de Previsões")
        forecast_prophet['ds'] = pd.to_datetime(forecast_prophet['ds'])
        tabela_filtrada = forecast_prophet[['ds', 'yhat']].copy()
        tabela_filtrada.columns = ["Data Projetada", "Valor Previsto"]
        st.dataframe(tabela_filtrada)
