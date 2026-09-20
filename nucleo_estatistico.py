import math

class EstatisticaDescritiva:
    """
    Núcleo estatístico próprio implementado 'na unha'.
    Todas as funções aqui devem ser validadas contra NumPy/SciPy posteriormente.
    """

    @staticmethod
    def _validar_dados(dados: list) -> None:
        """Método auxiliar para validação básica dos dados."""
        if not dados:
            raise ValueError("A lista de dados não pode estar vazia.")
        if not all(isinstance(x, (int, float)) for x in dados):
            raise TypeError("Todos os elementos devem ser numéricos (int ou float).")

    @staticmethod
    def media(dados: list[float]) -> float:
        """Calcula a média aritmética."""
        EstatisticaDescritiva._validar_dados(dados)
        return sum(dados) / len(dados)

    @staticmethod
    def mediana(dados: list[float]) -> float:
        """Calcula a mediana (valor central)."""
        EstatisticaDescritiva._validar_dados(dados)
        dados_ordenados = sorted(dados)
        n = len(dados_ordenados)
        meio = n // 2
        
        if n % 2 == 0:
            # Se for par, a mediana é a média dos dois valores centrais
            return (dados_ordenados[meio - 1] + dados_ordenados[meio]) / 2.0
        else:
            # Se for ímpar, a mediana é o valor do meio
            return float(dados_ordenados[meio])

    @staticmethod
    def moda(dados: list[float]) -> list[float]:
        """
        Calcula a moda (valor mais frequente). 
        Retorna uma lista, pois pode haver mais de uma moda (multimodal).
        """
        EstatisticaDescritiva._validar_dados(dados)
        frequencias = {}
        for valor in dados:
            frequencias[valor] = frequencias.get(valor, 0) + 1
        
        max_frequencia = max(frequencias.values())
        
        # Se todos os valores aparecem apenas uma vez, não há moda
        if max_frequencia == 1:
            return [] 
            
        return [valor for valor, freq in frequencias.items() if freq == max_frequencia]

    @staticmethod
    def amplitude(dados: list[float]) -> float:
        """Calcula a amplitude (max - min)."""
        EstatisticaDescritiva._validar_dados(dados)
        return max(dados) - min(dados)

    @staticmethod
    def variancia(dados: list[float], populacional: bool = False) -> float:
        """
        Calcula a variância amostral (padrão) ou populacional.
        populacional=True divide por N, caso contrário divide por N-1.
        """
        EstatisticaDescritiva._validar_dados(dados)
        n = len(dados)
        
        if not populacional and n < 2:
            raise ValueError("A variância amostral requer pelo menos 2 elementos.")
            
        media_dados = EstatisticaDescritiva.media(dados)
        soma_quadrados = sum((x - media_dados) ** 2 for x in dados)
        
        denominador = n if populacional else n - 1
        return soma_quadrados / denominador

    @staticmethod
    def desvio_padrao(dados: list[float], populacional: bool = False) -> float:
        """Calcula o desvio padrão (raiz quadrada da variância)."""
        var = EstatisticaDescritiva.variancia(dados, populacional)
        return math.sqrt(var)

    @staticmethod
    def quartis(dados: list[float]) -> tuple[float, float, float]:
        """
        Calcula os quartis (Q1, Q2, Q3).
        Método: Divisão da lista ordenada em metades.
        """
        EstatisticaDescritiva._validar_dados(dados)
        dados_ordenados = sorted(dados)
        n = len(dados_ordenados)
        q2 = EstatisticaDescritiva.mediana(dados_ordenados)
        
        if n % 2 == 0:
            metade_inferior = dados_ordenados[:n//2]
            metade_superior = dados_ordenados[n//2:]
        else:
            # Se ímpar, exclui a mediana (Q2) para calcular Q1 e Q3
            metade_inferior = dados_ordenados[:n//2]
            metade_superior = dados_ordenados[n//2 + 1:]
            
        q1 = EstatisticaDescritiva.mediana(metade_inferior)
        q3 = EstatisticaDescritiva.mediana(metade_superior)
        
        return q1, q2, q3

    @staticmethod
    def coeficiente_variacao(dados: list[float]) -> float:
        """Calcula o coeficiente de variação (CV) em porcentagem."""
        media_dados = EstatisticaDescritiva.media(dados)
        if media_dados == 0:
            raise ValueError("A média é zero, o coeficiente de variação é indefinido.")
        desvio = EstatisticaDescritiva.desvio_padrao(dados)
        return (desvio / media_dados) * 100.0

    @staticmethod
    def covariancia(x: list[float], y: list[float]) -> float:
        """Calcula a covariância amostral entre duas listas."""
        if len(x) != len(y):
            raise ValueError("As listas x e y devem ter o mesmo tamanho.")
        EstatisticaDescritiva._validar_dados(x)
        EstatisticaDescritiva._validar_dados(y)
        
        n = len(x)
        if n < 2:
            raise ValueError("A covariância amostral requer pelo menos 2 pares de dados.")
            
        media_x = EstatisticaDescritiva.media(x)
        media_y = EstatisticaDescritiva.media(y)
        
        soma_produtos = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))
        return soma_produtos / (n - 1)

    @staticmethod
    def correlacao_pearson(x: list[float], y: list[float]) -> float:
        """Calcula o coeficiente de correlação de Pearson (r)."""
        cov = EstatisticaDescritiva.covariancia(x, y)
        desvio_x = EstatisticaDescritiva.desvio_padrao(x)
        desvio_y = EstatisticaDescritiva.desvio_padrao(y)
        
        if desvio_x == 0 or desvio_y == 0:
            raise ValueError("O desvio padrão é zero, a correlação é indefinida.")
            
        return cov / (desvio_x * desvio_y)