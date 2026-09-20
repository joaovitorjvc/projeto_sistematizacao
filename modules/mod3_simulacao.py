"""
Módulo 3 — Probabilidade e Simulação de Monte Carlo.

Dois experimentos obrigatórios:
  (a) Lei dos Grandes Números (LLN) — convergência da frequência relativa.
  (b) Teorema Central do Limite (TCL) — distribuição das médias amostrais.

OBS: A geração de números aleatórios usa numpy (não é estatística, é aleatoriedade).
     Os cálculos estatísticos (média, variância) vêm do nosso nucleo_estatistico.py.
"""
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from nucleo_estatistico import EstatisticaDescritiva
from utils.carregar_dados import obter_colunas_numericas


# ============================================================
# (a) LEI DOS GRANDES NÚMEROS
# ============================================================
def _experimento_lln():
    st.subheader("🎲 (a) Lei dos Grandes Números (LLN)")
    st.markdown(
        """
        A **Lei dos Grandes Números** afirma que, à medida que o número de
        repetições de um experimento aumenta, a **frequência relativa** de um
        evento se aproxima da sua **probabilidade teórica**.

        Vamos simular lançamentos de uma moeda (ou dado) e observar a convergência.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        tipo = st.selectbox("Experimento:", ["Moeda (cara/coroa)", "Dado (1 a 6)"])
    with col2:
        n_lancamentos = st.slider(
            "Número de lançamentos:", min_value=10, max_value=50000,
            value=5000, step=10,
        )
    with col3:
        semente = st.number_input("Semente aleatória (seed):", value=42, step=1)

    # Simulação
    rng = np.random.default_rng(int(semente))

    if "Moeda" in tipo:
        # 0 = coroa, 1 = cara. P(cara) = 0.5
        resultados = rng.integers(0, 2, size=n_lancamentos)
        prob_teorica = 0.5
        evento_label = "Frequência relativa de CARA"
        cor = "#4C72B0"
    else:
        # 1 a 6. P(sair 6) = 1/6
        resultados = rng.integers(1, 7, size=n_lancamentos)
        prob_teorica = 1 / 6
        evento_label = "Frequência relativa de sair 6"
        cor = "#C44E52"

    # Frequência relativa acumulada
    if "Moeda" in tipo:
        acertos_acumulados = np.cumsum(resultados == 1)
    else:
        acertos_acumulados = np.cumsum(resultados == 6)

    eixo_x = np.arange(1, n_lancamentos + 1)
    freq_relativa = acertos_acumulados / eixo_x

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eixo_x, freq_relativa, color=cor, linewidth=1.5, label="Frequência relativa")
    ax.axhline(
        prob_teorica, color="red", linestyle="--", linewidth=2,
        label=f"Probabilidade teórica = {prob_teorica:.4f}",
    )
    ax.set_xlabel("Número de lançamentos")
    ax.set_ylabel(evento_label)
    ax.set_title(f"Convergência da Frequência Relativa — {tipo}")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Métricas finais
    freq_final = freq_relativa[-1]
    st.info(
        f"**Após {n_lancamentos:,} lançamentos:**\n\n".replace(",", ".") +
        f"- Frequência relativa final: **{freq_final:.5f}**\n" +
        f"- Probabilidade teórica: **{prob_teorica:.5f}**\n" +
        f"- Diferença absoluta: **{abs(freq_final - prob_teorica):.5f}**"
    )


# ============================================================
# (b) TEOREMA CENTRAL DO LIMITE
# ============================================================
def _experimento_tcl(df):
    st.subheader("📊 (b) Teorema Central do Limite (TCL)")
    st.markdown(
        """
        O **Teorema Central do Limite** afirma que a distribuição das **médias
        amostrais** se aproxima de uma distribuição **Normal**, independentemente
        da distribuição original da variável (desde que o tamanho da amostra seja
        suficientemente grande).

        Vamos sortear amostras repetidas de uma variável do nosso dataset e
        observar o histograma das médias.
        """
    )

    colunas_num = obter_colunas_numericas(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        variavel = st.selectbox("Variável do dataset:", colunas_num)
    with col2:
        tam_amostra = st.slider("Tamanho de cada amostra (n):", 2, 500, 30, step=2)
    with col3:
        n_amostras = st.slider("Número de amostras repetidas:", 100, 5000, 1000, step=100)
    with col4:
        semente = st.number_input("Semente (seed):", value=42, step=1, key="seed_tcl")

    # Limpeza da variável
    serie = df[variavel].dropna().to_numpy()
    if len(serie) < tam_amostra:
        st.error("Dataset pequeno demais para o tamanho de amostra escolhido.")
        return

    # Amostragem
    rng = np.random.default_rng(int(semente))
    medias_amostrais = np.empty(n_amostras)
    for i in range(n_amostras):
        amostra = rng.choice(serie, size=tam_amostra, replace=True)
        # Usa o NOSSO núcleo estatístico para a média
        medias_amostrais[i] = EstatisticaDescritiva.media(amostra.tolist())

    # Estatísticas das médias amostrais (usando o núcleo próprio)
    media_das_medias = EstatisticaDescritiva.media(medias_amostrais.tolist())
    desvio_das_medias = EstatisticaDescritiva.desvio_padrao(medias_amostrais.tolist())

    # Média e desvio da variável original (para comparação teórica)
    media_original = EstatisticaDescritiva.media(serie.tolist())
    desvio_original = EstatisticaDescritiva.desvio_padrao(serie.tolist())

    # Desvio teórico esperado pelo TCL: sigma / sqrt(n)
    desvio_teorico = desvio_original / math.sqrt(tam_amostra)

    # Plot do histograma das médias + curva Normal teórica
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(
        medias_amostrais, bins=40, density=True,
        color="#55A868", edgecolor="white", alpha=0.8,
        label="Médias amostrais",
    )

    # Curva Normal teórica — implementada "na unha" com math
    x = np.linspace(media_das_medias - 4 * desvio_das_medias,
                    media_das_medias + 4 * desvio_das_medias, 300)
    y = [
        (1 / (desvio_das_medias * math.sqrt(2 * math.pi))) *
        math.exp(-((xi - media_das_medias) ** 2) / (2 * desvio_das_medias ** 2))
        for xi in x
    ]
    ax.plot(x, y, color="red", linewidth=2, label="Curva Normal ajustada")
    ax.set_title(f"Distribuição das Médias Amostrais — {variavel} (n={tam_amostra})")
    ax.set_xlabel("Média amostral")
    ax.set_ylabel("Densidade")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Métricas
    st.markdown("### 📌 Comparação com o TCL")
    c1, c2, c3 = st.columns(3)
    c1.metric("Média das médias", f"{media_das_medias:.4f}",
              help="Deve se aproximar da média original")
    c2.metric("Média original (dataset)", f"{media_original:.4f}")
    c3.metric("Desvio teórico (σ/√n)", f"{desvio_teorico:.4f}",
              help="Desvio esperado pela teoria do TCL")

    st.markdown(
        f"""
        - **Desvio padrão das médias observado:** `{desvio_das_medias:.4f}`
        - **Desvio padrão teórico (σ/√n):** `{desvio_teorico:.4f}`
        - **Diferença relativa:** `{abs(desvio_das_medias - desvio_teorico) / desvio_teorico * 100:.2f}%`
        """
    )


# ============================================================
# RENDER PRINCIPAL DO MÓDULO 3
# ============================================================
def render(df):
    st.header("🎲 Módulo 3 — Probabilidade e Simulação de Monte Carlo")
    st.markdown(
        """
        Este módulo utiliza **simulação de Monte Carlo** para demonstrar dois
        teoremas fundamentais da estatística. Os parâmetros são controláveis
        pelo usuário, permitindo observar os fenômenos em ação.
        """
    )
    st.divider()

    _experimento_lln()
    st.divider()
    _experimento_tcl(df)