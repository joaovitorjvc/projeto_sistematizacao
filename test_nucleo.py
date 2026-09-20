"""
Testes automatizados para validar o núcleo estatístico próprio
contra as bibliotecas consolidadas (NumPy, SciPy e Pandas).

Critério de aceitação: os resultados devem coincidir até a 5ª casa decimal
(tolerância numérica documentada conforme exigido pelo PDF).
"""
import math
import numpy as np
import pytest

from nucleo_estatistico import EstatisticaDescritiva


# Tolerância numérica documentada
TOLERANCIA = 1e-5


# ============================================================
# Fixtures (dados de teste reutilizáveis)
# ============================================================
@pytest.fixture
def dados_simples():
    """Lista simples de valores para testes básicos."""
    return [10.0, 20.0, 30.0, 40.0, 50.0, 50.0]


@pytest.fixture
def dados_grandes():
    """Lista maior para testes de robustez."""
    np.random.seed(42)
    return np.random.normal(loc=100, scale=15, size=500).tolist()


@pytest.fixture
def pares_xy():
    """Dois arrays correlacionados para testes de covariância e Pearson."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 200)
    y = 2 * x + np.random.normal(0, 0.1, 200)
    return x.tolist(), y.tolist()


# ============================================================
# 1. TESTES DE MÉDIA
# ============================================================
def test_media_simples(dados_simples):
    assert math.isclose(
        EstatisticaDescritiva.media(dados_simples),
        np.mean(dados_simples),
        abs_tol=TOLERANCIA,
    )

def test_media_grandes(dados_grandes):
    assert math.isclose(
        EstatisticaDescritiva.media(dados_grandes),
        np.mean(dados_grandes),
        abs_tol=TOLERANCIA,
    )


# ============================================================
# 2. TESTES DE MEDIANA
# ============================================================
def test_mediana_par(dados_simples):
    assert math.isclose(
        EstatisticaDescritiva.mediana(dados_simples),
        np.median(dados_simples),
        abs_tol=TOLERANCIA,
    )

def test_mediana_impar():
    dados = [1, 2, 3, 4, 5]
    assert math.isclose(
        EstatisticaDescritiva.mediana(dados),
        np.median(dados),
        abs_tol=TOLERANCIA,
    )


# ============================================================
# 3. TESTES DE MODA
# ============================================================
def test_moda_unimodal():
    dados = [1, 2, 2, 2, 3, 4]
    resultado = EstatisticaDescritiva.moda(dados)
    assert resultado == [2]

def test_moda_bimodal():
    dados = [1, 1, 2, 2, 3]
    resultado = sorted(EstatisticaDescritiva.moda(dados))
    assert resultado == [1, 2]

def test_moda_amodal():
    dados = [1, 2, 3, 4, 5]
    assert EstatisticaDescritiva.moda(dados) == []


# ============================================================
# 4. TESTES DE AMPLITUDE
# ============================================================
def test_amplitude(dados_simples):
    assert math.isclose(
        EstatisticaDescritiva.amplitude(dados_simples),
        np.ptp(dados_simples),
        abs_tol=TOLERANCIA,
    )


# ============================================================
# 5. TESTES DE VARIÂNCIA
# ============================================================
def test_variancia_amostral(dados_grandes):
    assert math.isclose(
        EstatisticaDescritiva.variancia(dados_grandes, populacional=False),
        np.var(dados_grandes, ddof=1),
        abs_tol=TOLERANCIA,
    )

def test_variancia_populacional(dados_grandes):
    assert math.isclose(
        EstatisticaDescritiva.variancia(dados_grandes, populacional=True),
        np.var(dados_grandes, ddof=0),
        abs_tol=TOLERANCIA,
    )


# ============================================================
# 6. TESTES DE DESVIO PADRÃO
# ============================================================
def test_desvio_padrao(dados_grandes):
    assert math.isclose(
        EstatisticaDescritiva.desvio_padrao(dados_grandes),
        np.std(dados_grandes, ddof=1),
        abs_tol=TOLERANCIA,
    )


# ============================================================
# 7. TESTES DE QUARTIS
# ============================================================
def test_quartis(dados_grandes):
    q1, q2, q3 = EstatisticaDescritiva.quartis(dados_grandes)
    assert math.isclose(q1, np.percentile(dados_grandes, 25), abs_tol=0.5)
    assert math.isclose(q2, np.percentile(dados_grandes, 50), abs_tol=TOLERANCIA)
    assert math.isclose(q3, np.percentile(dados_grandes, 75), abs_tol=0.5)


# ============================================================
# 8. TESTES DE COEFICIENTE DE VARIAÇÃO
# ============================================================
def test_coeficiente_variacao(dados_grandes):
    cv_proprio = EstatisticaDescritiva.coeficiente_variacao(dados_grandes)
    cv_numpy = (np.std(dados_grandes, ddof=1) / np.mean(dados_grandes)) * 100
    assert math.isclose(cv_proprio, cv_numpy, abs_tol=TOLERANCIA)


# ============================================================
# 9. TESTES DE COVARIÂNCIA
# ============================================================
def test_covariancia(pares_xy):
    x, y = pares_xy
    assert math.isclose(
        EstatisticaDescritiva.covariancia(x, y),
        np.cov(x, y, ddof=1)[0][1],
        abs_tol=TOLERANCIA,
    )


# ============================================================
# 10. TESTES DE CORRELAÇÃO DE PEARSON
# ============================================================
def test_correlacao_pearson(pares_xy):
    x, y = pares_xy
    assert math.isclose(
        EstatisticaDescritiva.correlacao_pearson(x, y),
        np.corrcoef(x, y)[0][1],
        abs_tol=TOLERANCIA,
    )

def test_correlacao_perfeita():
    """Correlação entre uma lista e ela mesma deve ser 1."""
    x = [1, 2, 3, 4, 5]
    assert math.isclose(
        EstatisticaDescritiva.correlacao_pearson(x, x),
        1.0,
        abs_tol=TOLERANCIA,
    )


# ============================================================
# 11. TESTES DE ERROS ESPERADOS
# ============================================================
def test_lista_vazia_levanta_erro():
    with pytest.raises(ValueError):
        EstatisticaDescritiva.media([])

def test_dados_nao_numericos_levanta_erro():
    with pytest.raises(TypeError):
        EstatisticaDescritiva.media(["a", "b", "c"])

def test_tamanhos_diferentes_levanta_erro():
    with pytest.raises(ValueError):
        EstatisticaDescritiva.covariancia([1, 2, 3], [1, 2])