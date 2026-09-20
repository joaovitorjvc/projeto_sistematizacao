"""
Módulo de carregamento e pré-processamento dos dados.
IMPORTANTE: Este módulo faz parte da camada de INTERFACE/UTILITÁRIOS.
As estatísticas exibidas na aplicação devem vir do nucleo_estatistico.py.
"""
import pandas as pd
import streamlit as st


# Caminho relativo ao arquivo CSV
CAMINHO_CSV = "data/Sales_transactions_2022_2025.csv"


@st.cache_data(show_spinner=False)
def carregar_dados(caminho: str = CAMINHO_CSV) -> pd.DataFrame:
    """
    Carrega o CSV e aplica uma limpeza básica.
    O cache do Streamlit evita recarregar a cada interação.
    """
    df = pd.read_csv(caminho)

    # 1. Padronizar strings: remover espaços extras e ajustar capitalização
    colunas_texto = df.select_dtypes(include="object").columns
    for col in colunas_texto:
        df[col] = df[col].astype(str).str.strip()
        # Substituir 'nan' (string) por valor nulo real
        df[col] = df[col].replace({"nan": None, "None": None, "": None})

    # 2. Padronizar gênero (male/Male/FEMALE/Non-binary/Non Binary)
    if "Customer_Gender" in df.columns:
        df["Customer_Gender"] = df["Customer_Gender"].str.capitalize()
        df["Customer_Gender"] = df["Customer_Gender"].replace({
            "Non Binary": "Non-binary",
            "Non-Binary": "Non-binary",
        })

    # 3. Padronizar status e flag de retorno
    if "Order_Status" in df.columns:
        df["Order_Status"] = df["Order_Status"].str.capitalize()
    if "Return_Flag" in df.columns:
        df["Return_Flag"] = df["Return_Flag"].str.capitalize()

    # 4. Converter Order_Date para datetime
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

    # 5. Garantir tipos numéricos corretos
    colunas_numericas = [
        "Customer_Age", "Quantity", "Unit_Price", "Discount_Percentage",
        "Sales_Amount", "Cost_Amount", "Profit", "Delivery_Days",
        "Customer_Rating", "Inventory_Level",
    ]
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def obter_colunas_numericas(df: pd.DataFrame) -> list:
    """Retorna a lista de colunas numéricas do DataFrame."""
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def obter_colunas_categoricas(df: pd.DataFrame) -> list:
    """Retorna a lista de colunas categóricas (texto) do DataFrame."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()