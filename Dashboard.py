import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoARIMA

import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="IPEA Dashboard", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Analise de Dados IPEA (Preço do Petróleo)")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

os.chdir(r"C:\Protheus\Curso\Analise IPEA\Analise_IPEA")
df_ipeadata = pd.read_csv('data_frame_ipeadata.csv', sep=';')
df_arima = pd.read_csv('data_frame_ipeadata.csv', sep=';', parse_dates=['data'])

col1, col2 = st.columns((2))
df_ipeadata["data"] = pd.to_datetime(df_ipeadata["data"])

# Getting the min and max date 
startDate = pd.to_datetime(df_ipeadata["data"]).min()
endDate = pd.to_datetime(df_ipeadata["data"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Data Inicio", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("Data Final", endDate))

df_ipeadata = df_ipeadata[(df_ipeadata["data"] >= date1) & (df_ipeadata["data"] <= date2)].copy()

with col1:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Plot do Preço do Petróleo")
    plt.plot(df_ipeadata.data, df_ipeadata.preco_petroleo)
    st.pyplot(use_container_width=True)

df_ipeadata.set_index('data', inplace=True)
df_ipeadata.sort_index(inplace= True)
period = int(len(df_ipeadata)/2)

resultados = seasonal_decompose(df_ipeadata, period = period, model = 'additive')

with col2:
    st.subheader("Decompondo o Data Frame utilizando um perido de 5 dias")
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize = (15,10))

    resultados.observed.plot(ax=ax1)
    resultados.trend.plot(ax=ax2)
    resultados.seasonal.plot(ax=ax3)
    resultados.resid.plot(ax=ax4)
    st.pyplot(fig, use_container_width=True)

cl1, cl2 = st.columns((2))
X = df_ipeadata.preco_petroleo.values
result = adfuller(X)

st.write("Teste ADF")
st.write(f"Teste Estatístico: {result[0]}")
st.write(f"P-Value: {result[1]}")
st.write("Valores críticos:")

for key, value in result[4].items():
  st.write(f"\t{key}: {value}")

with cl1:
    ma = df_ipeadata.rolling(12).mean()
    st.subheader("Execução do Augmented Dickey Fuller")
    f, ax = plt.subplots()
    df_ipeadata.plot(ax=ax, legend=False)
    ma.plot(ax=ax, legend=False, color='r')
    plt.tight_layout()
    st.pyplot(f)
 
with cl2:
    st.subheader("Execução do Augmented Dickey Fuller")
    df_log = np.log(df_ipeadata)
    ma_log = df_log.rolling(12).mean()

    f, ax = plt.subplots()
    df_log.plot(ax=ax, legend=False)
    ma_log.plot(ax=ax, legend=False, color='r')
    st.pyplot(f)
        
cl3, cl4 = st.columns((2))

with cl3:
    df_s = (df_log - ma_log).dropna()

    ma_s = df_s.rolling(12).mean()

    std = df_s.rolling(12).std()

    st.subheader("Execução do Augmented Dickey Fuller")
    f, ax = plt.subplots()
    df_s.plot(ax=ax, legend=False)
    ma_s.plot(ax=ax, legend=False, color='r')
    std.plot(ax=ax, legend=False, color='g')
    plt.tight_layout()
    st.pyplot(f)

X_s = df_s.preco_petroleo.values
result_s = adfuller(X_s)

with cl4:
    st.subheader("Execução do Augmented Dickey Fuller")
    df_diff = df_s.diff(1)
    ma_diff = df_diff.rolling(12).mean()

    std_diff = df_diff.rolling(12).std()


    f, ax = plt.subplots()
    df_diff.plot(ax=ax, legend=False)
    ma_diff.plot(ax=ax, legend=False, color='r')
    std_diff.plot(ax=ax, legend=False, color='g')
    plt.tight_layout()

    st.pyplot(f)

X_diff = df_diff.preco_petroleo.dropna().values
result_diff = adfuller(X_diff)

cl5, cl6 = st.columns((2))

lag_acf = acf(df_diff.dropna(), nlags=25)

with cl5:
    st.subheader("1.96/sqrt(N-d) -> N - número de pontos do df e d é o número de vezes que nós diferenciamos o df")
    plt.plot(lag_acf)
    plt.axhline(y= -1.96/(np.sqrt((len(df_diff) -1))), linestyle='--', color='gray',linewidth=0.7)
    plt.axhline(y=0, linestyle='--', color='gray',linewidth=0.7)
    plt.axhline(y= 1.96/(np.sqrt((len(df_diff) -1))), linestyle='--', color='gray',linewidth=0.7)
    plt.title("ACF")
    st.pyplot()
    
lag_pacf = pacf(df_diff.dropna(), nlags=25)

with cl6:
    st.subheader("1.96/sqrt(N-d) -> N - número de pontos do df e d é o número de vezes que nós diferenciamos o df")
    plt.plot(lag_pacf)
    plt.axhline(y= -1.96/(np.sqrt((len(df_diff) -1))), linestyle='--', color='gray',linewidth=0.7)
    plt.axhline(y=0, linestyle='--', color='gray',linewidth=0.7)
    plt.axhline(y= 1.96/(np.sqrt((len(df_diff) -1))), linestyle='--', color='gray',linewidth=0.7)
    plt.title("PACF")
    st.pyplot()

cl7, cl8 = st.columns((2))

with cl7:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Verificando a Auto correlaçao")
    fig = plot_acf(df_ipeadata.preco_petroleo)
    st.pyplot(fig, use_container_width=True)
    
with cl8:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Verificando a Auto correlação parcial")
    fig = plot_pacf(df_ipeadata.preco_petroleo)
    st.pyplot(fig, use_container_width=True)


df_arima["preco_petroleo"] = df_arima["preco_petroleo"].astype(float)
df_arima['id'] = 'ipea'
df_arima = df_arima.sort_values(by='data').reset_index(drop=True)
df_1 = df_arima.rename(columns={'data':'ds','preco_petroleo':'y','id':'unique_id'})

treino = df_1.loc[df_1['ds'] < '2023-06-01']
valid = df_1.loc[(df_1['ds'] >= '2023-06-01') & (df_1['ds'] < '2024-01-01')]
h = valid['ds'].nunique()

def wmape(y_true, y_pred):
  return np.abs(y_true-y_pred).sum() / np.abs(y_true).sum()

# 0. Naive

model = StatsForecast(models=[Naive()], freq='D', n_jobs=-1)
model.fit(treino)

forecast_df_naive = model.predict(h=h, level=[90])
forecast_df_naive = forecast_df_naive.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

wmape_naive = wmape(forecast_df_naive['y'].values, forecast_df_naive['Naive'].values)

st.subheader("Utilizando o metodo Naive para efetuar a previsão do modelo de teste")
fig = model.plot(treino, forecast_df_naive, level=[90], unique_ids=['ipea'],engine ='matplotlib', max_insample_length=90)
st.pyplot(fig)
st.dataframe(forecast_df_naive, use_container_width=True)
st.write(f"WMAPE: {wmape_naive:.2%}")

# 1. AutoARIMA

model = StatsForecast(models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
model.fit(treino)

forecast_dfArima = model.predict(h=h, level=[90])
forecast_dfArima = forecast_dfArima.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

wmape1 = wmape(forecast_dfArima['y'].values, forecast_dfArima['AutoARIMA'].values)

st.subheader("Utilizando o metodo AutoARIMA para efetuar a previsão do modelo de teste")
fig = model.plot(treino, forecast_dfArima, level=[90], unique_ids=['ipea'],engine ='matplotlib', max_insample_length=90)
st.pyplot(fig)
st.dataframe(forecast_dfArima,use_container_width=True)
st.write(f"WMAPE: {wmape1:.2%}")

# 2. Seasonal Naive

model_s = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
model_s.fit(treino)

forecast_dfs = model_s.predict(h=h, level=[90])
forecast_dfs = forecast_dfs.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

wmape2 = wmape(forecast_dfs['y'].values, forecast_dfs['SeasonalNaive'].values)
print(f"WMAPE: {wmape2:.2%}")

st.subheader("Utilizando o metodo SeasonalNaive para efetuar a previsão do modelo teste")
fig = model_s.plot(treino, forecast_dfs, level=[90], unique_ids=['ipea'],engine ='matplotlib', max_insample_length=120)
st.pyplot(fig)
st.dataframe(forecast_dfs,use_container_width=True)
st.write(f"WMAPE: {wmape2:.2%}")


# 4. Completo

model_a = StatsForecast(models=[Naive(),AutoARIMA(season_length=7), SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
model_a.fit(treino)

forecast_dfa = model_a.predict(h=h, level=[90])
forecast_dfa = forecast_dfa.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

wmape4 = wmape(forecast_dfa['y'].values, forecast_dfa['AutoARIMA'].values)
print(f"WMAPE: {wmape4:.2%}")

st.subheader("Utilizando os metodos Naive, autoARIMA e SeasonalNaive para efetuar a previsão do modelo teste")
fig = model_a.plot(treino, forecast_dfa, level=[90], unique_ids=['ipea'],engine ='matplotlib', max_insample_length=90)
st.pyplot(fig)
st.dataframe(forecast_dfa,use_container_width=True)
st.write(f"WMAPE: {wmape4:.2%}")


st.write(f"WMAPE Naive: {wmape_naive:.2%}")
st.write(f"WMAPE AutoARIMA: {wmape1:.2%}")
st.write(f"WMAPE SeasonalNaive: {wmape2:.2%}")
st.write(f"WMAPE Naive, AutoARIMA e SeasonalNaive: {wmape4:.2%}")
