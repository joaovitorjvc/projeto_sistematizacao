"""Módulo 2 — Estatística Descritiva Interativa (usa o núcleo próprio)."""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from nucleo_estatistico import EstatisticaDescritiva
from utils.carregar_dados import obter_colunas_numericas


def render(df: pd.DataFrame):
    st.header("📈 Módulo 2 — Estatística Descritiva Interativa")
    st.markdown(
        "Selecione uma **variável numérica** para visualizar suas medidas de tendência central, "
        "dispersão e distribuição. Os valores abaixo são calculados pelo **nosso núcleo próprio**."
    )

    colunas_num = obter_colunas_numericas(df)
    if not colunas_num:
        st.warning("Nenhuma coluna numérica disponível.")
        return

    # Seleção da variável
    coluna = st.selectbox("Selecione a variável:", colunas_num)

    # Limpeza dos dados (remover NaN da coluna selecionada)
    serie = df[coluna].dropna()
    dados = serie.tolist()

    if len(dados) < 2:
        st.error("Dados insuficientes para análise (menos de 2 valores válidos).")
        return

    # ---- Medidas de tendência central ----
    st.subheader("🎯 Tendência Central")
    c1, c2, c3 = st.columns(3)
    c1.metric("Média", f"{EstatisticaDescritiva.media(dados):.2f}")
    c2.metric("Mediana", f"{EstatisticaDescritiva.mediana(dados):.2f}")
    modas = EstatisticaDescritiva.moda(dados)
    moda_str = ", ".join(f"{m:.2f}" for m in modas) if modas else "Sem moda (amodal)"
    c3.metric("Moda", moda_str)

    # ---- Medidas de dispersão ----
    st.subheader("📏 Dispersão")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amplitude", f"{EstatisticaDescritiva.amplitude(dados):.2f}")
    c2.metric("Variância (amostral)", f"{EstatisticaDescritiva.variancia(dados):.2f}")
    c3.metric("Desvio Padrão", f"{EstatisticaDescritiva.desvio_padrao(dados):.2f}")
    c4.metric("Coef. Variação", f"{EstatisticaDescritiva.coeficiente_variacao(dados):.2f}%")

    # ---- Quartis e Outliers (IQR) ----
    st.subheader("📦 Quartis e Outliers (Regra do IQR)")
    q1, q2, q3 = EstatisticaDescritiva.quartis(dados)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    c1, c2, c3 = st.columns(3)
    c1.metric("Q1", f"{q1:.2f}")
    c2.metric("Q2 (Mediana)", f"{q2:.2f}")
    c3.metric("Q3", f"{q3:.2f}")

    outliers = [x for x in dados if x < limite_inferior or x > limite_superior]
    st.info(
        f"**Limite inferior:** {limite_inferior:.2f} | "
        f"**Limite superior:** {limite_superior:.2f} | "
        f"**Outliers detectados:** {len(outliers)} "
        f"({100 * len(outliers) / len(dados):.2f}% dos dados)"
    )

    # ---- Visualizações ----
    st.subheader("📊 Visualizações")
    tab1, tab2 = st.tabs(["Histograma", "Boxplot"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(dados, bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Histograma — {coluna}")
        ax.set_xlabel(coluna)
        ax.set_ylabel("Frequência")
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.boxplot(dados, vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#55A868"))
        ax.set_title(f"Boxplot — {coluna}")
        ax.set_xlabel(coluna)
        st.pyplot(fig)
        plt.close(fig)

    # ---- Tabela de Frequência (para variáveis com poucos valores únicos) ----
    if serie.nunique() <= 30:
        st.subheader("📋 Tabela de Frequências")
        freq = serie.value_counts().sort_index().reset_index()
        freq.columns = [coluna, "Frequência Absoluta"]
        freq["Frequência Relativa (%)"] = (freq["Frequência Absoluta"] / len(serie) * 100).round(2)
        st.dataframe(freq, use_container_width=True)