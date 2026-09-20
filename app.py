"""
Aplicação principal do Laboratório Estatístico Interativo.
Disciplina: Matemática e Estatística para Computação
"""
import streamlit as st

from utils.carregar_dados import carregar_dados
from modules import mod0_dados, mod2_descritiva, mod3_simulacao, mod4_distribuicoes, mod5_correlacao, mod6_relatorio

# ---- Configuração da página ----
st.set_page_config(
    page_title="Laboratório Estatístico Interativo",
    page_icon="📊",
    layout="wide",
)

# ---- Sidebar ----
with st.sidebar:
    st.title("📊 Lab. Estatístico")
    st.markdown("**Matemática e Estatística para Computação**")
    st.divider()

    modulo = st.radio(
        "Navegação:",
        [
            "🏠 Módulo 0 — Dados",
            "📈 Módulo 2 — Descritiva",
            "🎲 Módulo 3 — Simulação",
            "📐 Módulo 4 — Distribuições",
            "🔗 Módulo 5 — Correlação",
            "📝 Módulo 6 — Descobertas",
        ],
    )

    st.divider()
    st.caption("Desenvolvido individualmente como projeto de Sistematização.")

# ---- Carregar dados uma única vez ----
try:
    df = carregar_dados()
except FileNotFoundError:
    st.error(
        "❌ Arquivo `data/Sales_transactions_2022_2025.csv` não encontrado. "
        "Coloque o CSV na pasta `data/`."
    )
    st.stop()

# ---- Roteamento dos módulos ----
if modulo == "🏠 Módulo 0 — Dados":
    mod0_dados.render(df)

elif modulo == "📈 Módulo 2 — Descritiva":
    mod2_descritiva.render(df)

elif modulo == "🎲 Módulo 3 — Simulação":
    from modules import mod3_simulacao
    mod3_simulacao.render(df)

elif modulo == "📐 Módulo 4 — Distribuições":
    mod4_distribuicoes.render(df)

elif modulo == "🔗 Módulo 5 — Correlação":
    mod5_correlacao.render(df)

elif modulo == "📝 Módulo 6 — Descobertas":
    mod6_relatorio.render(df)