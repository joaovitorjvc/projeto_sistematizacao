"""
Módulo 4 — Distribuições Teóricas.

Sobreponha ao histograma de uma variável do dataset uma curva de distribuição
teórica, com parâmetros estimados a partir dos dados.

Distribuições implementadas "na unha":
  - Normal (contínua)
  - Exponencial (contínua)
  - Uniforme (contínua)
  - Poisson (discreta)
"""
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from nucleo_estatistico import EstatisticaDescritiva
from utils.carregar_dados import obter_colunas_numericas


# ============================================================
# PDFs / PMFs "NA UNHA"
# ============================================================
def pdf_normal(x: float, mu: float, sigma: float) -> float:
    """Função densidade de probabilidade da Normal."""
    if sigma <= 0:
        return 0.0
    coef = 1.0 / (sigma * math.sqrt(2 * math.pi))
    expoente = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coef * math.exp(expoente)


def pdf_exponencial(x: float, lambd: float) -> float:
    """Função densidade de probabilidade da Exponencial. λ > 0, x >= 0."""
    if x < 0 or lambd <= 0:
        return 0.0
    return lambd * math.exp(-lambd * x)


def pdf_uniforme(x: float, a: float, b: float) -> float:
    """Função densidade de probabilidade da Uniforme em [a, b]."""
    if b <= a:
        return 0.0
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def pmf_poisson(k: int, lambd: float) -> float:
    """Função de probabilidade da Poisson. k >= 0 inteiro, λ > 0."""
    if k < 0 or lambd <= 0:
        return 0.0
    return (lambd ** k) * math.exp(-lambd) / math.factorial(k)


# ============================================================
# HELPERS
# ============================================================
DISTRIBUICOES = {
    "Normal": "Contínua — adequada para variáveis simétricas",
    "Exponencial": "Contínua — adequada para tempos/esperas",
    "Uniforme": "Contínua — adequada para variáveis sem pico central",
    "Poisson": "Discreta — adequada para contagens",
}


def _estimar_parametros_normal(dados: list[float]) -> tuple[float, float]:
    """Estima μ (média) e σ (desvio padrão amostral) a partir dos dados."""
    mu = EstatisticaDescritiva.media(dados)
    sigma = EstatisticaDescritiva.desvio_padrao(dados)
    return mu, sigma


def _estimar_parametros_exponencial(dados: list[float]) -> float:
    """Estima λ = 1/média para a Exponencial."""
    media = EstatisticaDescritiva.media(dados)
    return 1.0 / media if media > 0 else 1.0


def _estimar_parametros_uniforme(dados: list[float]) -> tuple[float, float]:
    """Estima a = min e b = max para a Uniforme."""
    return min(dados), max(dados)


def _estimar_parametros_poisson(dados: list[float]) -> float:
    """Estima λ = média dos dados para a Poisson."""
    return EstatisticaDescritiva.media(dados)


# ============================================================
# RENDER PRINCIPAL
# ============================================================
def render(df):
    st.header("📐 Módulo 4 — Distribuições Teóricas")
    st.markdown(
        """
        Escolha uma **variável numérica** e uma **distribuição teórica**.
        A curva será sobreposta ao histograma, com parâmetros estimados a
        partir dos próprios dados. Todas as PDFs/PMFs foram implementadas
        **"na unha"** com a biblioteca `math`.
        """
    )

    colunas_num = obter_colunas_numericas(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        variavel = st.selectbox("Variável:", colunas_num, index=0)
    with col2:
        dist_nome = st.selectbox(
            "Distribuição teórica:",
            list(DISTRIBUICOES.keys()),
            format_func=lambda x: f"{x} — {DISTRIBUICOES[x]}",
        )
    with col3:
        n_bins = st.slider("Nº de bins do histograma:", 10, 100, 40)

    # Limpeza
    serie = df[variavel].dropna()
    if len(serie) < 10:
        st.error("Dados insuficientes para ajuste (mínimo 10 valores).")
        return

    # Poisson requer dados inteiros não-negativos
    if dist_nome == "Poisson":
        if (serie < 0).any() or (serie != serie.astype(int)).any():
            st.warning(
                "⚠️ A distribuição de Poisson só faz sentido para variáveis "
                "**inteiras e não-negativas** (ex: `Quantity`, `Delivery_Days`)."
            )
            return

    dados = serie.tolist()

    # ---- Estimativa de parâmetros ----
    st.subheader("🔧 Parâmetros Estimados")
    if dist_nome == "Normal":
        mu, sigma = _estimar_parametros_normal(dados)
        st.markdown(
            f"- **μ (média):** `{mu:.4f}`\n"
            f"- **σ (desvio padrão):** `{sigma:.4f}`"
        )
    elif dist_nome == "Exponencial":
        lambd = _estimar_parametros_exponencial(dados)
        st.markdown(
            f"- **λ (taxa) = 1/média:** `{lambd:.6f}`\n"
            f"- **Média teórica = 1/λ:** `{1/lambd:.4f}`"
        )
    elif dist_nome == "Uniforme":
        a, b = _estimar_parametros_uniforme(dados)
        st.markdown(
            f"- **a (mínimo observado):** `{a:.4f}`\n"
            f"- **b (máximo observado):** `{b:.4f}`"
        )
    elif dist_nome == "Poisson":
        lambd = _estimar_parametros_poisson(dados)
        st.markdown(f"- **λ (média):** `{lambd:.4f}`")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Histograma em densidade
    ax.hist(
        dados, bins=n_bins, density=True,
        color="#4C72B0", edgecolor="white", alpha=0.75,
        label=f"Dados — {variavel}",
    )

    # Sobrepõe a curva teórica
    x_min, x_max = min(dados), max(dados)
    margem = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
    eixo_x = np.linspace(x_min - margem, x_max + margem, 500)

    if dist_nome == "Normal":
        ys = [pdf_normal(float(x), mu, sigma) for x in eixo_x]
        ax.plot(eixo_x, ys, color="red", linewidth=2.5,
                label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")

    elif dist_nome == "Exponencial":
        ys = [pdf_exponencial(float(x), lambd) for x in eixo_x if x >= 0]
        eixo_x_pos = [float(x) for x in eixo_x if x >= 0]
        ax.plot(eixo_x_pos, ys, color="red", linewidth=2.5,
                label=f"Exponencial(λ={lambd:.4f})")

    elif dist_nome == "Uniforme":
        ys = [pdf_uniforme(float(x), a, b) for x in eixo_x]
        ax.plot(eixo_x, ys, color="red", linewidth=2.5,
                label=f"Uniforme(a={a:.2f}, b={b:.2f})")

    elif dist_nome == "Poisson":
        # Para Poisson, plotamos as PMFs nos valores inteiros observados
        k_max = int(max(dados))
        ks = np.arange(0, k_max + 1)
        # Normaliza a PMF pelo intervalo (para ficar comparável ao histograma em densidade)
        pmfs = np.array([pmf_poisson(int(k), lambd) for k in ks])
        # Mostra como gráfico de barras sobreposto (stem-like)
        ax.bar(ks, pmfs, width=0.6, color="red", alpha=0.4,
               label=f"Poisson(λ={lambd:.2f})")
        # Linha conectando os pontos
        ax.plot(ks, pmfs, "o-", color="red", linewidth=1.5, markersize=5)

    ax.set_xlabel(variavel)
    ax.set_ylabel("Densidade")
    ax.set_title(f"Ajuste da distribuição {dist_nome} — {variavel}")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # ---- Discussão da qualidade do ajuste ----
    st.subheader("🔍 Qualidade do Ajuste")
    _discutir_ajuste(dados, dist_nome, variavel)


# ============================================================
# DISCUSSÃO AUTOMÁTICA DA QUALIDADE DO AJUSTE
# ============================================================
def _discutir_ajuste(dados: list[float], dist_nome: str, variavel: str):
    """Compara descritivamente as estatísticas dos dados com as esperadas pela teoria."""
    media = EstatisticaDescritiva.media(dados)
    desvio = EstatisticaDescritiva.desvio_padrao(dados)
    q1, mediana, q3 = EstatisticaDescritiva.quartis(dados)
    iqr = q3 - q1
    assimetria_aprox = (media - mediana) / desvio if desvio > 0 else 0
    cv = EstatisticaDescritiva.coeficiente_variacao(dados)

    st.markdown(
        f"""
        **Métricas dos dados:**
        - Média = `{media:.4f}` | Mediana = `{mediana:.4f}` | Desvio padrão = `{desvio:.4f}`
        - Q1 = `{q1:.4f}` | Q3 = `{q3:.4f}` | IQR = `{iqr:.4f}`
        - Coef. de variação (CV) = `{cv:.2f}%`
        - Assimetria empírica ≈ `{assimetria_aprox:.4f}` (média−mediana)/desvio
        """
    )

    if dist_nome == "Normal":
        if abs(assimetria_aprox) < 0.1:
            st.success(
                "✅ **Ajuste visualmente adequado.** A média e a mediana estão próximas, "
                "sugerindo simetria consistente com a Normal."
            )
        elif abs(assimetria_aprox) < 0.3:
            st.warning(
                "⚠️ **Ajuste razoável.** Há leve assimetria. Ajuste pode melhorar com "
                "transformação dos dados (ex: log) ou escolhendo outra distribuição."
            )
        else:
            st.error(
                "❌ **Ajuste inadequado.** Forte assimetria detectada. A distribuição "
                "Normal provavelmente não é a melhor escolha para esta variável."
            )

    elif dist_nome == "Exponencial":
        # Exponencial teórica tem CV = 100% e média ≈ desvio padrão
        if abs(cv - 100) < 20:
            st.success(
                "✅ **Ajuste visualmente adequado.** O coeficiente de variação está "
                "próximo de 100%, consistente com a Exponencial."
            )
        else:
            st.warning(
                f"⚠️ **Ajuste parcial.** O CV observado é `{cv:.2f}%` "
                "enquanto o esperado para a Exponencial é ~100%."
            )

    elif dist_nome == "Uniforme":
        # Uniforme tem CV = (b-a) / (sqrt(12) * média) — esperamos um CV relativamente baixo
        st.info(
            "ℹ️ **Análise da Uniforme.** Esta distribuição tem densidade constante "
            "entre `a` e `b`. Se o histograma mostrar um pico central, a Uniforme "
            "não será um bom ajuste."
        )

    elif dist_nome == "Poisson":
        # Poisson tem média = variância
        variancia = EstatisticaDescritiva.variancia(dados)
        razao = variancia / media if media > 0 else float("inf")
        if abs(razao - 1.0) < 0.2:
            st.success(
                f"✅ **Ajuste visualmente adequado.** A razão variância/média = "
                f"`{razao:.4f}` está próxima de 1, como esperado para Poisson."
            )
        else:
            st.warning(
                f"⚠️ **Ajuste parcial.** A razão variância/média = `{razao:.4f}`. "
                "Para a Poisson, espera-se ≈ 1 (equidispersão)."
            )