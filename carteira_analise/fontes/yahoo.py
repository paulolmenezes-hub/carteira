"""Camada de acesso a dados via Yahoo Finance (yfinance).

Esta é a ÚNICA camada do pacote que faz I/O de rede. Mantê-la isolada dos
módulos de cálculo (``tecnicos.py``, ``fundamentalistas.py``) é o que torna
o resto do pacote testável sem rede e o que permitiria, no futuro, trocar
ou complementar a fonte de dados (ex.: brapi.dev — ver Seção 8.2.1 do
manual) escrevendo apenas um novo módulo aqui, sem tocar na lógica de
análise.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def baixar_precos(ticker: str, periodo: str) -> pd.Series:
    """Baixa o histórico de preços de fechamento de um ticker.

    Retorna uma pandas.Series vazia se o ticker não existir ou não houver
    dados — quem chama decide o que fazer (o motor de análise trata isso
    como "dados insuficientes", não como erro fatal).
    """
    hist = yf.download(ticker, period=periodo, progress=False)
    if hist.empty:
        return pd.Series(dtype=float)

    fechamento = hist["Close"]
    if isinstance(fechamento, pd.DataFrame):
        fechamento = fechamento.squeeze(axis=1)
    return fechamento.dropna()


def baixar_info(ticker: str) -> dict:
    """Baixa o dicionário 'info' bruto do Yahoo Finance para um ticker."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def baixar_dividendos(ticker: str) -> pd.Series:
    """Baixa o histórico de dividendos/rendimentos pagos de um ticker."""
    try:
        divs = yf.Ticker(ticker).dividends
    except Exception:
        return pd.Series(dtype=float)
    return divs if divs is not None else pd.Series(dtype=float)
