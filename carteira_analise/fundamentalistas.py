"""Indicadores fundamentalistas e de dividendos.

Assim como em ``tecnicos.py``, este módulo é deliberadamente livre de I/O de
rede: recebe estruturas já buscadas (o dicionário ``info`` do Yahoo Finance,
ou a série de dividendos pagos) e devolve indicadores derivados. Isso torna
a lógica testável com dados sintéticos e reutilizável caso, no futuro, outra
fonte de dados (ex.: brapi.dev — ver Seção 8.2.1 do manual) seja adotada:
basta escrever um adaptador que gere as mesmas estruturas de entrada.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def extrair_fundamentalistas_acao(info: dict) -> dict:
    """Extrai os múltiplos relevantes de ações do dicionário 'info' do
    Yahoo Finance. Não usamos o campo 'dividendYield' bruto de propósito:
    em vários tickers da B3 ele vem com escala inconsistente (às vezes já
    em %, às vezes fração), gerando valores absurdos como '1224%'. O yield
    real é calculado à parte, em `calcular_historico_dividendos`, a partir
    dos dividendos efetivamente pagos."""
    return {
        "pl": info.get("trailingPE"),
        "pvp": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "divida_patrimonio": info.get("debtToEquity"),
    }


def extrair_fundamentalistas_fii(info: dict) -> dict:
    """Extrai os múltiplos relevantes de FIIs (mesma ressalva sobre
    dividendYield bruto — ver docstring de extrair_fundamentalistas_acao)."""
    return {
        "pvp": info.get("priceToBook"),
        "volume_medio": info.get("averageVolume"),
    }


@dataclass
class HistoricoDividendos:
    soma_12m: float
    soma_anterior_12m: float
    crescimento_yoy: float | None
    dy_12m_real: float | None
    n_pagamentos_12m: int


def calcular_historico_dividendos(
    dividendos: pd.Series, preco_atual: float
) -> HistoricoDividendos | None:
    """Calcula yield efetivo (12m), crescimento ano contra ano e
    regularidade dos pagamentos, a partir do histórico REAL de dividendos
    pagos — mais confiável que o campo 'dividendYield' isolado da API, que
    só reflete um instante e tem escala inconsistente entre tickers.

    `dividendos` deve ser uma pandas.Series indexada por data (timezone
    naive), com o valor pago em cada evento.
    """
    if dividendos is None or dividendos.empty:
        return None

    if dividendos.index.tz is not None:
        dividendos = dividendos.copy()
        dividendos.index = dividendos.index.tz_localize(None)

    hoje = dividendos.index.max()
    ult_12m = dividendos[dividendos.index > hoje - pd.Timedelta(days=365)]
    anterior_12m = dividendos[
        (dividendos.index <= hoje - pd.Timedelta(days=365))
        & (dividendos.index > hoje - pd.Timedelta(days=730))
    ]

    soma_12m = float(ult_12m.sum())
    soma_anterior_12m = float(anterior_12m.sum())

    crescimento_yoy = None
    if soma_anterior_12m > 0:
        crescimento_yoy = (soma_12m - soma_anterior_12m) / soma_anterior_12m

    dy_12m_real = (soma_12m / preco_atual) if preco_atual else None

    return HistoricoDividendos(
        soma_12m=soma_12m,
        soma_anterior_12m=soma_anterior_12m,
        crescimento_yoy=crescimento_yoy,
        dy_12m_real=dy_12m_real,
        n_pagamentos_12m=int(len(ult_12m)),
    )
