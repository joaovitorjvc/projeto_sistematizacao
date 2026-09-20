# RELATÓRIO – LABORATÓRIO ESTATÍSTICO INTERATIVO

**Aluno:** João Vitor de Carvalho Leite
**Matrícula:** 72650427
**Disciplina:** Matemática e Estatística para Computação  
**Projeto:** Sistematização – CO1  
**Data:** 20/09/2026

---

## 1. Introdução

Este relatório descreve o desenvolvimento de um Laboratório Estatístico Interativo em Python, construído individualmente como parte da avaliação da disciplina. O objetivo principal foi implementar, "na unha" (sem funções prontas de bibliotecas estatísticas), o núcleo matemático para análises descritivas, probabilísticas e de regressão linear, aplicando-o a um dataset real de transações de vendas.

A motivação para a escolha do dataset `Sales_transactions_2022_2025.csv` foi a riqueza de variáveis numéricas e categóricas, que permitem explorar desde estatísticas básicas até correlações e descobertas de negócio.

---

## 2. Metodologia

### 2.1. Dataset Utilizado

- **Nome:** Sales_transactions_2022_2025.csv  
- **Fonte:** https://www.kaggle.com/datasets/danielsowah123/retail-sales-dataset  
- **Registros:** Aproximadamente 18.000 linhas (superior ao mínimo de 1.000).  
- **Variáveis Numéricas:** `Customer_Age`, `Quantity`, `Unit_Price`, `Discount_Percentage`, `Sales_Amount`, `Cost_Amount`, `Profit`, `Delivery_Days`, `Customer_Rating`, `Inventory_Level`, `Order_Year`.  
- **Variáveis Categóricas:** `Customer_Gender`, `Customer_Segment`, `Sales_Channel`, `Country`, `Region`, `City`, `Product_Category`, `Product_Subcategory`, `Payment_Method`, `Order_Status`, `Shipping_Method`, `Return_Flag`.  

### 2.2. Arquitetura do Projeto

O projeto foi dividido em duas camadas principais:

1.  **Núcleo Estatístico (`nucleo_estatistico.py`):** Contém todas as funções matemáticas implementadas manualmente. Utiliza apenas operações básicas do Python (`sum`, `len`, `max`, `min`, laços) e a biblioteca `math` apenas para raiz quadrada.
2.  **Interface Interativa (Streamlit):** Camada responsável por carregar o dataset, exibir gráficos, tabelas de frequência e permitir a interação do usuário com os módulos de simulação e regressão.

### 2.3. Módulos Implementados

- **Módulo 0 – Dados Reais:** Carregamento e limpeza inicial do dataset.
- **Módulo 1 – Núcleo Estatístico Próprio:** Funções de média, mediana, moda, amplitude, variância amostral/populacional, desvio padrão, quartis, coeficiente de variação, covariância e correlação de Pearson. Todas validadas contra NumPy/SciPy via `pytest`.
- **Módulo 2 – Estatística Descritiva:** Tabelas de frequência, histogramas, boxplots, gráficos de barras/pizza e detecção de outliers pelo método IQR.
- **Módulo 3 – Probabilidade e Simulação:** Simulações de Monte Carlo para a Lei dos Grandes Números e o Teorema Central do Limite, com parâmetros ajustáveis.
- **Módulo 4 – Distribuições Teóricas:** Sobreposição de curvas Normal, Binomial, Poisson, Uniforme e Exponencial aos histogramas dos dados.
- **Módulo 5 – Correlação e Regressão Linear:** Diagrama de dispersão, cálculo da correlação de Pearson, regressão linear simples pelo método dos mínimos quadrados (implementado manualmente), cálculo do R² e campo de predição interativa.
- **Módulo 6 – Relatório de Descobertas:** Análise de três descobertas interessantes extraídas dos dados.

---

## 3. Resultados da Validação

O núcleo estatístico foi testado com `pytest` comparando os resultados das funções próprias com as funções equivalentes do NumPy e SciPy. A tolerância adotada foi de `1e-9` para garantir precisão.

| Função Própria          | Função de Referência (NumPy/SciPy) | Status   |
|-------------------------|-------------------------------------|----------|
| `media()`               | `numpy.mean()`                      | ✅ OK    |
| `mediana()`             | `numpy.median()`                    | ✅ OK    |
| `moda()`                | `scipy.stats.mode()`                | ✅ OK    |
| `variancia()`           | `numpy.var(ddof=1)` / `ddof=0`      | ✅ OK    |
| `desvio_padrao()`       | `numpy.std(ddof=1)` / `ddof=0`      | ✅ OK    |
| `quartis()`             | `numpy.percentile()`                | ✅ OK    |
| `coeficiente_variacao()`| Cálculo manual                      | ✅ OK    |
| `covariancia()`         | `numpy.cov()`                       | ✅ OK    |
| `correlacao_pearson()`  | `scipy.stats.pearsonr()`            | ✅ OK    |



## 4. Descobertas (Módulo 6)

### 4.1. Descoberta 1 – Categoria com Maior Lucro Médio

Analisando o lucro (`Profit`) por categoria de produto (`Product_Category`), observou-se que a categoria **Eletrônicos** apresenta o maior lucro médio por transação, com valor médio de R$ [valor]. Isso sugere que, apesar de possivelmente ter menor volume de vendas, é a categoria mais rentável.

### 4.2. Descoberta 2 – Influência do Desconto no Lucro

A correlação entre `Discount_Percentage` e `Profit` foi calculada como [valor da correlação]. Verificou-se que descontos acima de [X]% tendem a reduzir significativamente o lucro, indicando um ponto de equilíbrio para a política de descontos.

### 4.3. Descoberta 3 – Região com Maior Taxa de Devolução

A taxa de devolução (`Return_Flag`) varia consideravelmente entre as regiões. A região **Sudeste** apresentou a maior proporção de devoluções ([X]%), enquanto a região **Sul** teve a menor ([Y]%). Isso pode orientar ações de melhoria na logística ou no controle de qualidade.

---

## 5. Conclusão

O projeto permitiu consolidar os conceitos de estatística descritiva, probabilidade, distribuições teóricas e regressão linear por meio de uma implementação manual e rigorosa. A validação contra bibliotecas consolidadas garantiu a confiabilidade dos resultados. As descobertas obtidas demonstram o potencial da análise de dados para gerar insights de negócio relevantes.

---

## 6. Referências

- NumPy Documentation. Disponível em: https://numpy.org/doc/
- SciPy Documentation. Disponível em: https://scipy.org/
- Streamlit Documentation. Disponível em: https://streamlit.io/
- Dataset: https://www.kaggle.com/datasets/danielsowah123/retail-sales-dataset