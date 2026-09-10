"""carteira_analise — motor de análise de carteiras de ações e FIIs.

Extraído do notebook original como parte do Horizonte 1 do roadmap de
evolução (Seção 8.1 do manual): separa o "motor de análise" (este pacote)
da camada de apresentação (o notebook, e futuramente uma aplicação web).

Uso típico:

    from carteira_analise import carteira
    from carteira_analise.fontes import yahoo

    resultados = carteira.analisar_carteira(
        tickers=["PETR4.SA", "VALE3.SA"],
        tipo="acao",
        periodo="2y",
        fonte=yahoo,
    )
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("carteira-analise")
except PackageNotFoundError:  # pacote rodando direto do código-fonte, sem instalação
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
