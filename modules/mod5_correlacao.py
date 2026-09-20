"""
Módulo 5 — Correlação e Regressão Linear Simples.

A regressão linear é implementada "NA UNHA" pelo método dos mínimos quadrados.
Não utilizamos np.polyfit, scipy.stats.linregress ou sklearn.

Fórmulas:
    β₁ = Σ[(xi - x̄)(yi - ȳ)] / Σ(xi - x̄)²
    β₀ = ȳ - β₁ · x̄
    R² = 1 - (SQ_res / SQ_total)
"""
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from nucleo_estatistico import EstatisticaDescritiva
from utils.carregar_dados import obter_colunas_numericas


# ============================================================
# REGRESSÃO LINEAR "NA UNHA" (MÍNIMOS QUADRADOS)
# ============================================================
class RegressaoLinear:
    """
    Implementação própria da regressão linear simples pelo método
    dos mínimos quadrados ordinários (OLS).
    """

    def __init__(self):
        self.beta_0 = None   # intercepto
        self.beta_1 = None   # coeficiente angular
        self.r2 = None
        self.n = 0
        self.x_media = None
        self.y_media = None

    def ajustar(self, x: list[float], y: list[float]) -> None:
        """Ajusta o modelo aos dados x (preditor) e y (resposta)."""
        if len(x) != len(y):
            raise ValueError("x e y devem ter o mesmo tamanho.")
        if len(x) < 2:
            raise ValueError("São necessários ao menos 2 pontos para o ajuste.")

        self.n = len(x)
        self.x_media = EstatisticaDescritiva.media(x)
        self.y_media = EstatisticaDescritiva.media(y)

        # Numerador: Σ[(xi - x̄)(yi - ȳ)]
        numerador = sum((xi - self.x_media) * (yi - self.y_media)
                        for xi, yi in zip(x, y))

        # Denominador: Σ(xi - x̄)²
        denominador = sum((xi - self.x_media) ** 2 for xi in x)

        if denominador == 0:
            raise ValueError("Variância de x é zero — regressão indefinida.")

        self.beta_1 = numerador / denominador
        self.beta_0 = self.y_media - self.beta_1 * self.x_media

        # R² (coeficiente de determinação)
        sq_total = sum((yi - self.y_media) ** 2 for yi in y)
        sq_res = sum((yi - self.prever(xi)) ** 2 for xi, yi in zip(x, y))
        self.r2 = 1.0 - (sq_res / sq_total) if sq_total > 0 else 0.0

    def prever(self, x: float) -> float:
        """Retorna ŷ = β₀ + β₁ · x."""
        if self.beta_0 is None or self.beta_1 is None:
            raise ValueError("Modelo ainda não foi ajustado.")
        return self.beta_0 + self.beta_1 * x

    def equacao(self) -> str:
        """Retorna a equação da reta como string legível."""
        sinal = "+" if self.beta_0 >= 0 else "-"
        return f"ŷ = {self.beta_1:.4f}·x {sinal} {abs(self.beta_0):.4f}"


# ============================================================
# RENDER PRINCIPAL DO MÓDULO 5
# ============================================================
def render(df):
    st.header("🔗 Módulo 5 — Correlação e Regressão Linear")
    st.markdown(
        """
        Selecione duas variáveis numéricas. Vamos:
        1. Calcular o **coeficiente de correlação de Pearson** (com o nosso núcleo).
        2. Ajustar uma **reta de regressão linear** pelo método dos **mínimos quadrados**
           (implementado na mão).
        3. Calcular o **R²** e permitir **predição interativa**.
        """
    )

    colunas_num = obter_colunas_numericas(df)
    if len(colunas_num) < 2:
        st.error("O dataset precisa de pelo menos 2 colunas numéricas.")
        return

    col1, col2 = st.columns(2)
    with col1:
        var_x = st.selectbox("Variável independente (X):", colunas_num, index=0)
    with col2:
        var_y = st.selectbox("Variável dependente (Y):", colunas_num, index=1)

    if var_x == var_y:
        st.warning("Selecione variáveis diferentes.")
        return

    # Limpeza dos dados
    sub = df[[var_x, var_y]].dropna()
    x = sub[var_x].tolist()
    y = sub[var_y].tolist()

    if len(x) < 10:
        st.error("Dados insuficientes (mínimo 10 pontos).")
        return

    # ---- Cálculo da correlação (núcleo próprio) ----
    r = EstatisticaDescritiva.correlacao_pearson(x, y)

    # ---- Ajuste da regressão (nossa implementação) ----
    modelo = RegressaoLinear()
    try:
        modelo.ajustar(x, y)
    except ValueError as e:
        st.error(f"Erro ao ajustar regressão: {e}")
        return

    # ---- Métricas principais ----
    st.subheader("📊 Resultados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correlação de Pearson (r)", f"{r:.4f}")
    c2.metric("R²", f"{modelo.r2:.4f}")
    c3.metric("Coef. angular (β₁)", f"{modelo.beta_1:.4f}")
    c4.metric("Intercepto (β₀)", f"{modelo.beta_0:.4f}")

    # Interpretação automática da força da correlação
    forca = _classificar_correlacao(r)
    st.info(f"**Interpretação:** correlação **{forca}** "
            f"(|r| = {abs(r):.4f}) entre `{var_x}` e `{var_y}`.")

    st.markdown(f"**Equação da reta ajustada:** `{modelo.equacao()}`")

    # ---- Gráfico de dispersão + reta ----
    st.subheader("📈 Diagrama de Dispersão com Reta Ajustada")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, alpha=0.35, s=15, color="#4C72B0", label="Dados observados")

    x_min, x_max = min(x), max(x)
    x_reta = np.linspace(x_min, x_max, 200)
    y_reta = [modelo.prever(float(xi)) for xi in x_reta]
    ax.plot(x_reta, y_reta, color="red", linewidth=2.5,
            label=f"Reta ajustada: {modelo.equacao()}")

    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.set_title(f"Regressão Linear — {var_y} em função de {var_x}")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # ---- Predição interativa ----
    st.subheader("🔮 Campo de Predição Interativa")
    st.markdown(
        f"Informe um valor de **{var_x}** e obteremos uma estimativa de **{var_y}** "
        f"usando a reta de regressão."
    )

    x_input = st.number_input(
        f"Valor de {var_x}:",
        min_value=float(min(x)),
        max_value=float(max(x)),
        value=float(EstatisticaDescritiva.media(x)),
        step=float((max(x) - min(x)) / 100 if max(x) > min(x) else 1.0),
    )
    y_pred = modelo.prever(float(x_input))

    # Intervalo de confiança empírico via desvio dos resíduos
    residuos = [yi - modelo.prever(xi) for xi, yi in zip(x, y)]
    desvio_residuos = EstatisticaDescritiva.desvio_padrao(residuos)
    margem = 1.96 * desvio_residuos  # ~95% (assumindo normalidade dos resíduos)

    st.success(
        f"**Previsão para {var_x} = {x_input:.2f}:** "
        f"{var_y} ≈ **{y_pred:.4f}** "
        f"(IC ~95% empírico: [{y_pred - margem:.4f}, {y_pred + margem:.4f}])"
    )

    # ---- Discussão honesta ----
    st.subheader("⚠️ Lembrete Estatístico")
    st.warning(
        """
        **Correlação não implica causalidade.**

        Mesmo com r alto, não podemos afirmar que uma variável *causa* a outra.
        Pode haver:
        - **Variável de confusão** (uma terceira variável influenciando ambas);
        - **Coincidência** ou relação espúria;
        - **Relação reversa** (Y causando X em vez do contrário).

        A regressão descreve a **associação linear** observada nos dados.
        """
    )

    # ---- Detalhamento do ajuste ----
    with st.expander("🔬 Detalhes do Ajuste"):
        sq_total = sum((yi - modelo.y_media) ** 2 for yi in y)
        sq_res = sum((yi - modelo.prever(xi)) ** 2 for xi, yi in zip(x, y))
        st.markdown(
            f"""
            - **Número de pontos (n):** `{modelo.n}`
            - **Média de X (x̄):** `{modelo.x_media:.4f}`
            - **Média de Y (ȳ):** `{modelo.y_media:.4f}`
            - **SQ total (Σ(yᵢ - ȳ)²):** `{sq_total:.4f}`
            - **SQ residual (Σ(yᵢ - ŷᵢ)²):** `{sq_res:.4f}`
            - **Desvio padrão dos resíduos:** `{desvio_residuos:.4f}`
            - **R² = 1 - SQ_res/SQ_total:** `{modelo.r2:.6f}`
            """
        )


# ============================================================
# CLASSIFICAÇÃO QUALITATIVA DA CORRELAÇÃO
# ============================================================
def _classificar_correlacao(r: float) -> str:
    """Classificação descritiva de |r| segundo critérios usuais."""
    a = abs(r)
    if a >= 0.9:
        return "muito forte"
    elif a >= 0.7:
        return "forte"
    elif a >= 0.5:
        return "moderada"
    elif a >= 0.3:
        return "fraca"
    else:
        return "muito fraca ou inexistente"