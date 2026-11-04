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

_SEGMENTOS_PAPEL = ("titulos", "val. mob", "val mob", "valores mobiliarios", "papel", "recebiveis",
                    "fundo de fundos")
# siglas curtas só valem como PALAVRA inteira: "cri" dentro de "escritorios"
# fazia fundos de escritórios (HGRE11, PVBI11) serem tratados como papel
_SIGLAS_PAPEL = ("cri", "cra", "fof")


def segmento_de_papel(seg: str) -> bool:
    import re
    return (any(chave in seg for chave in _SEGMENTOS_PAPEL)
            or any(re.search(rf"\b{sigla}\b", seg) for sigla in _SIGLAS_PAPEL))


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
        if segmento_de_papel(seg):
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


# ---------------------------------------------------------------------------
# Regularidade dos dividendos (ações): DY por ano civil nos 5 anos fechados
# ---------------------------------------------------------------------------

ANOS_REGULARIDADE = 5
FRACAO_ANO_FRACO = 0.5        # ano "fraco": menos da metade da mediana dos 5 anos
FATOR_ANO_EXTRAORDINARIO = 2  # ano "fora da curva": mais que o dobro da mediana


def _sem_fuso_serie(serie: pd.Series | None) -> pd.Series:
    if serie is None:
        return pd.Series(dtype=float)
    serie = serie.dropna()
    if getattr(serie.index, "tz", None) is not None:
        serie = serie.copy()
        serie.index = serie.index.tz_localize(None)
    return serie.sort_index()


def dy_por_ano(dividendos: pd.Series | None, precos: pd.Series | None, anos) -> dict[int, float | None]:
    """Dividend yield de cada ano civil: proventos com data no ano ÷ preço
    médio de fechamento do ano (menos sensível que o preço de 31/12 a um
    pregão atípico). None se o ano não tiver preços suficientes."""
    d, p = _sem_fuso_serie(dividendos), _sem_fuso_serie(precos)
    saida: dict[int, float | None] = {}
    for ano in anos:
        precos_ano = p[p.index.year == ano]
        if len(precos_ano) < 150 or precos_ano.mean() <= 0:  # ~150 de ~248 pregões: ano coberto
            saida[ano] = None
            continue
        saida[ano] = float(d[d.index.year == ano].sum()) / float(precos_ano.mean())
    return saida


def regularidade_dividendos(dividendos: pd.Series | None, precos: pd.Series | None,
                            data_ref=None) -> dict | None:
    """Os dividendos de uma ação são regulares? Olha o DY de cada um dos 5
    anos civis FECHADOS antes de ``data_ref`` (o ano em curso, incompleto,
    sempre pareceria fraco: aparece à parte, "até agora").

    - Regular: pagou nos 5 anos e o pior ano rendeu pelo menos metade da
      mediana dos 5;
    - Oscilante: pagou nos 5, mas algum ano rendeu menos da metade da mediana;
    - Irregular: ficou pelo menos 1 ano sem pagar.

    Independente do selo, marca os anos "fora da curva" (mais que o dobro da
    mediana, em geral provento extraordinário). None sem preços dos 5 anos ou
    sem nenhum provento registrado (dado ausente, não "não pagou").

    Teste (Seção 14 do notebook, 04/11/2026; ~40 ações, 143 casos de 2022 a
    2025): com o selo Regular, a renda se manteve no ano seguinte em 99% dos
    casos, contra 82% nos Oscilantes e Irregulares (+17 p.p.; IC 90% +6 a
    +28). A mediana NÃO previu melhor o DY do ano seguinte que a média ou o
    último ano (erro médio de 3,8, 3,4 e 3,3 p.p.) — por isso o texto não
    recomenda olhar a mediana; ela segue só como régua do ano "fraco"."""
    d, p = _sem_fuso_serie(dividendos), _sem_fuso_serie(precos)
    if p.empty or d.empty:
        return None
    ref = pd.Timestamp(data_ref) if data_ref is not None else p.index[-1]
    anos = list(range(ref.year - ANOS_REGULARIDADE, ref.year))
    dys = dy_por_ano(d, p, anos)
    if any(v is None for v in dys.values()):
        return None
    valores = [dys[a] for a in anos]
    serie = pd.Series(valores, index=anos)
    mediana = float(serie.median())
    pagou = int((serie > 0).sum())
    pior_ano = int(serie.idxmin())
    pior = float(serie.min())
    extraordinarios = [a for a in anos if mediana > 0 and dys[a] > FATOR_ANO_EXTRAORDINARIO * mediana]
    if pagou < ANOS_REGULARIDADE:
        selo = "irregular"
    elif pior < FRACAO_ANO_FRACO * mediana:
        selo = "oscilante"
    else:
        selo = "regular"
    corrente = None
    p_ano = p[(p.index.year == ref.year) & (p.index <= ref)]
    if len(p_ano):
        dy_parcial = float(d[(d.index.year == ref.year) & (d.index <= ref)].sum()) / float(p_ano.mean())
        corrente = {"ano": ref.year, "dy": dy_parcial, "ate": ref}
    return {"anos": [{"ano": a, "dy": dys[a]} for a in anos], "pagou": pagou, "n_anos": ANOS_REGULARIDADE,
            "pior": pior, "ano_pior": pior_ano, "mediana": mediana, "media": float(serie.mean()),
            "extraordinarios": extraordinarios, "selo": selo, "ano_corrente": corrente}


def texto_regularidade(reg: dict) -> str:
    """Frase simples do bloco "Os dividendos são regulares?"."""
    def pct(v):
        return f"{v * 100:.1f}%".replace(".", ",")
    sem = reg["n_anos"] - reg["pagou"]
    if reg["selo"] == "irregular":
        texto = (f"Ficou {sem} {'anos' if sem > 1 else 'ano'} sem pagar dividendos nos últimos {reg['n_anos']}. "
                 "Não conte com essa renda.")
    elif reg["selo"] == "oscilante":
        texto = (f"Pagou todos os anos, mas em {reg['ano_pior']} rendeu só {pct(reg['pior'])}, menos da metade "
                 f"do normal dela ({pct(reg['mediana'])}).")
    else:
        texto = (f"Pagou todos os anos e o pior ano ({pct(reg['pior'])}) ainda ficou perto do normal dela "
                 f"({pct(reg['mediana'])}).")
    if reg["extraordinarios"]:
        anos = reg["extraordinarios"]
        dy = {a["ano"]: a["dy"] for a in reg["anos"]}
        lista = ", ".join(f"{a} ({pct(dy[a])})" for a in anos)
        texto += (f" Atenção: {lista} {'foram' if len(anos) > 1 else 'foi'} fora da curva, provavelmente com "
                  "provento extraordinário — não conte que se repita no mesmo nível.")
    return texto
