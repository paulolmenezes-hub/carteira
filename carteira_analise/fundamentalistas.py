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
    dividendos: pd.Series, preco_atual: float, data_ref=None
) -> HistoricoDividendos | None:
    """Calcula yield efetivo (12m), crescimento ano contra ano e
    regularidade dos pagamentos, a partir do histórico REAL de dividendos
    pagos — mais confiável que o campo 'dividendYield' isolado da API, que
    só reflete um instante e tem escala inconsistente entre tickers.

    `dividendos` deve ser uma pandas.Series indexada por data (timezone
    naive), com o valor pago em cada evento. `data_ref`: fim da janela de
    12 meses (padrão: hoje).
    """
    if dividendos is None or dividendos.empty:
        return None

    if dividendos.index.tz is not None:
        dividendos = dividendos.copy()
        dividendos.index = dividendos.index.tz_localize(None)

    # Os 12 meses contam a partir de HOJE (ou da data de referência), não do
    # último pagamento: um ativo que parou de pagar há anos mostrava o yield
    # daquela época sobre o preço atual (caso MRVE3: 13,8%, "acima do CDI").
    hoje = pd.Timestamp(data_ref) if data_ref is not None else pd.Timestamp.today().normalize()
    if hoje.tzinfo is not None:
        hoje = hoje.tz_localize(None)
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


# ---------------------------------------------------------------------------
# Tipo de FII: papel, tijolo ou híbrido
# ---------------------------------------------------------------------------

_SEGMENTOS_PAPEL = ("titulos", "val. mob", "val mob", "valores mobiliarios", "papel", "recebiveis", "cri",
                    "fundo de fundos", "fof")


def _segmento_normalizado(fund: dict | None) -> str:
    import unicodedata
    seg = str((fund or {}).get("segmento") or "")
    return "".join(c for c in unicodedata.normalize("NFD", seg) if unicodedata.category(c) != "Mn").lower().strip()


def segmento_generico(fund: dict | None) -> bool:
    """Segmento que não diz se o FII é de papel ou de tijolo."""
    seg = _segmento_normalizado(fund)
    return (not seg) or seg in ("outros", "nan", "none", "-") or "multicategoria" in seg


def classificar_fii(fund: dict | None) -> str | None:
    """'papel', 'tijolo', 'hibrido' ou None (sem dado), pelo SEGMENTO que o
    Fundamentus informa. A versão anterior olhava só a quantidade de imóveis:
    um fundo de recebíveis que recebe um imóvel em pagamento de uma dívida
    (caso RECR11) passava a ser tratado como tijolo, com "vacância de 100%".
    Com segmento genérico ("Multicategoria", "Outros") ou sem segmento, decide
    a participação dos imóveis no patrimônio (``imoveis_pl``: abaixo de 20%,
    papel; de 20% a 50%, híbrido; acima, tijolo) — o RECR11 aparece no
    Fundamentus como "Multicategoria", com 1 imóvel que é 3,2% do patrimônio.
    Sem esse dado, vale a quantidade de imóveis."""
    fund = fund or {}
    seg = _segmento_normalizado(fund)
    generico = segmento_generico(fund)
    if not generico:
        if "hibrido" in seg:
            return "hibrido"
        if any(chave in seg for chave in _SEGMENTOS_PAPEL):
            return "papel"
        return "tijolo"
    # segmento genérico ("Multicategoria", "Outros"): decide a participação dos
    # imóveis no patrimônio (Imóveis/PL, página de detalhes do Fundamentus)
    imoveis_pl = fund.get("imoveis_pl")
    if imoveis_pl is not None:
        if imoveis_pl < 0.20:
            return "papel"
        return "hibrido" if imoveis_pl < 0.50 else "tijolo"
    qtd = fund.get("qtd_imoveis")
    if qtd in (None, ""):
        return None
    return "tijolo" if qtd not in (0, 0.0) else "papel"


def tem_imoveis(fund: dict | None) -> bool:
    """Vacância e cap rate só se aplicam a FIIs de tijolo ou híbridos."""
    return classificar_fii(fund) in ("tijolo", "hibrido")
