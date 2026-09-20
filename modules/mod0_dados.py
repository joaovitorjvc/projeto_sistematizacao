"""Módulo 0 — Apresentação dos Dados Reais."""
import streamlit as st
import pandas as pd


def render(df: pd.DataFrame):
    st.header("📊 Módulo 0 — Conhecendo os Dados")
    st.markdown(
        """
        Este laboratório utiliza um dataset público de **transações de vendas** entre 2022 e 2025,
        cobrindo múltiplas regiões, categorias de produtos e canais de venda.
        """
    )

    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Registros", f"{len(df):,}".replace(",", "."))
    col2.metric("Total de Colunas", df.shape[1])
    col3.metric("Colunas Numéricas", len(df.select_dtypes(include="number").columns))
    col4.metric("Colunas Categóricas", len(df.select_dtypes(include="object").columns))

    st.divider()

    # Amostra dos dados
    st.subheader("🔍 Amostra do Dataset")
    st.dataframe(df.head(20), use_container_width=True)

    # Estatísticas rápidas
    st.subheader("📋 Resumo Estrutural")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    # Valores nulos
    st.subheader("⚠️ Valores Nulos por Coluna")
    nulos = df.isnull().sum().sort_values(ascending=False)
    nulos = nulos[nulos > 0]
    if len(nulos) > 0:
        st.dataframe(nulos.rename("Qtd. Nulos").to_frame(), use_container_width=True)
    else:
        st.success("Nenhum valor nulo encontrado.")