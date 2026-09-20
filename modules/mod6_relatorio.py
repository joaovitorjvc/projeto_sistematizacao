"""
Módulo 6 — Relatório de Descobertas.

Gera automaticamente um painel com as principais descobertas extraídas
do dataset, para embasar o relatório escrito.
"""
import streamlit as st
import matplotlib.pyplot as plt

from nucleo_estatistico import EstatisticaDescritiva
from utils.carregar_dados import obter_colunas_numericas, obter_colunas_categoricas


def render(df):
    st.header("📝 Módulo 6 — Relatório de Descobertas")
    st.markdown(
        """
        Esta página reúne as **3 descobertas mais interessantes** extraídas do
        dataset. Os números abaixo foram calculados com o **nosso núcleo
        estatístico próprio** e podem ser usados diretamente no relatório.
        """
    )

    # ========================================================
    # DESCOBERTA 1 — Categoria mais lucrativa
    # ========================================================
    st.subheader("💡 Descoberta 1 — Qual categoria é mais lucrativa?")
    _descoberta_categoria(df)

    st.divider()

    # ========================================================
    # DESCOBERTA 2 — Impacto do desconto no lucro
    # ========================================================
    st.subheader("💡 Descoberta 2 — O desconto afeta o lucro?")
    _descoberta_desconto(df)

    st.divider()

    # ========================================================
    # DESCOBERTA 3 — Canal de vendas vs. ticket médio
    # ========================================================
    st.subheader("💡 Descoberta 3 — Qual canal tem o maior ticket médio?")
    _descoberta_canal(df)


# ============================================================
# DESCOBERTA 1: Categoria mais lucrativa
# ============================================================
def _descoberta_categoria(df):
    sub = df[["Product_Category", "Profit"]].dropna()

    # Agrupar por categoria
    resumo = []
    for categoria in sub["Product_Category"].unique():
        valores = sub[sub["Product_Category"] == categoria]["Profit"].tolist()
        if len(valores) < 5:
            continue
        resumo.append({
            "Categoria": categoria,
            "N": len(valores),
            "Lucro Médio": EstatisticaDescritiva.media(valores),
            "Lucro Total": sum(valores),
            "Desvio Padrão": EstatisticaDescritiva.desvio_padrao(valores),
        })

    if not resumo:
        st.warning("Dados insuficientes.")
        return

    import pandas as pd
    df_resumo = pd.DataFrame(resumo).sort_values("Lucro Médio", ascending=False)

    # Formatação
    df_show = df_resumo.copy()
    for col in ["Lucro Médio", "Lucro Total", "Desvio Padrão"]:
        df_show[col] = df_show[col].apply(lambda v: f"R$ {v:,.2f}")
    st.dataframe(df_show, use_container_width=True)

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_resumo["Categoria"], df_resumo["Lucro Médio"], color="#4C72B0")
    ax.set_ylabel("Lucro Médio (R$)")
    ax.set_title("Lucro Médio por Categoria")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    st.pyplot(fig)
    plt.close(fig)

    melhor = df_resumo.iloc[0]
    st.success(
        f"📌 **Descoberta:** A categoria **{melhor['Categoria']}** tem o maior "
        f"lucro médio (R$ {melhor['Lucro Médio']:,.2f}) com {melhor['N']} transações."
    )


# ============================================================
# DESCOBERTA 2: Impacto do desconto no lucro
# ============================================================
def _descoberta_desconto(df):
    sub = df[["Discount_Percentage", "Profit"]].dropna()

    # Agrupar por faixa de desconto
    faixas = {
        "0%": sub[sub["Discount_Percentage"] == 0]["Profit"].tolist(),
        "1–10%": sub[(sub["Discount_Percentage"] > 0) & (sub["Discount_Percentage"] <= 10)]["Profit"].tolist(),
        "11–20%": sub[(sub["Discount_Percentage"] > 10) & (sub["Discount_Percentage"] <= 20)]["Profit"].tolist(),
        ">20%": sub[sub["Discount_Percentage"] > 20]["Profit"].tolist(),
    }

    resumo = []
    for faixa, valores in faixas.items():
        if len(valores) >= 5:
            resumo.append({
                "Faixa de Desconto": faixa,
                "N": len(valores),
                "Lucro Médio": EstatisticaDescritiva.media(valores),
            })

    if not resumo:
        st.warning("Dados insuficientes.")
        return

    import pandas as pd
    df_resumo = pd.DataFrame(resumo)

    df_show = df_resumo.copy()
    df_show["Lucro Médio"] = df_show["Lucro Médio"].apply(lambda v: f"R$ {v:,.2f}")
    st.dataframe(df_show, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_resumo["Faixa de Desconto"], df_resumo["Lucro Médio"], color="#C44E52")
    ax.set_ylabel("Lucro Médio (R$)")
    ax.set_title("Lucro Médio por Faixa de Desconto")
    ax.grid(alpha=0.3, axis="y")
    st.pyplot(fig)
    plt.close(fig)

    # Correlação
    r = EstatisticaDescritiva.correlacao_pearson(
        sub["Discount_Percentage"].tolist(),
        sub["Profit"].tolist(),
    )
    st.success(
        f"📌 **Descoberta:** Correlação entre desconto e lucro: "
        f"**r = {r:.4f}** — sugere que descontos maiores tendem a "
        f"{'reduzir' if r < 0 else 'aumentar'} o lucro médio."
    )


# ============================================================
# DESCOBERTA 3: Ticket médio por canal
# ============================================================
def _descoberta_canal(df):
    sub = df[["Sales_Channel", "Sales_Amount"]].dropna()

    resumo = []
    for canal in sub["Sales_Channel"].unique():
        valores = sub[sub["Sales_Channel"] == canal]["Sales_Amount"].tolist()
        if len(valores) >= 5:
            resumo.append({
                "Canal": canal,
                "N": len(valores),
                "Ticket Médio": EstatisticaDescritiva.media(valores),
                "Desvio Padrão": EstatisticaDescritiva.desvio_padrao(valores),
            })

    if not resumo:
        st.warning("Dados insuficientes.")
        return

    import pandas as pd
    df_resumo = pd.DataFrame(resumo).sort_values("Ticket Médio", ascending=False)

    df_show = df_resumo.copy()
    df_show["Ticket Médio"] = df_show["Ticket Médio"].apply(lambda v: f"R$ {v:,.2f}")
    df_show["Desvio Padrão"] = df_show["Desvio Padrão"].apply(lambda v: f"R$ {v:,.2f}")
    st.dataframe(df_show, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_resumo["Canal"], df_resumo["Ticket Médio"], color="#55A868")
    ax.set_ylabel("Ticket Médio (R$)")
    ax.set_title("Ticket Médio por Canal de Venda")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    st.pyplot(fig)
    plt.close(fig)

    melhor = df_resumo.iloc[0]
    st.success(
        f"📌 **Descoberta:** O canal **{melhor['Canal']}** tem o maior ticket médio "
        f"(R$ {melhor['Ticket Médio']:,.2f}) com {melhor['N']} transações."
    )