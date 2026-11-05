"""Cotações históricas oficiais da B3 (arquivos COTAHIST), resumidas por mês.

Por que existe (05/11/2026, caso AFHI11): o Yahoo Finance só tem cotações de
alguns papéis a partir de uma data recente (AFHI11: 01/2024, embora negocie
desde 2021), e sem o preço dos meses antigos o yield desses meses não pode ser
calculado. O Fundamentus mostra só a cotação AJUSTADA por proventos, que não
serve para o yield. A B3 publica o fechamento real de todos os papéis, um
arquivo por ano (dezenas de MB) — pesado demais para o painel baixar a cada
análise. Por isso o notebook (Seção 15) baixa os anos encerrados uma vez e
gera um resumo mensal pequeno, ``dados/cotacoes_b3_mensais.csv.gz``, que
fica no repositório: fechamento do último pregão e média dos fechamentos de
cada mês, por papel (ações, FIIs, ETFs e units do mercado à vista).

Os preços da B3 NÃO são ajustados por desdobramentos/grupamentos; o Yahoo
ajusta. ``completar_precos`` põe a base na escala do Yahoo com os eventos
que o próprio Yahoo registra, antes de juntar as duas séries.

Layout oficial (registro tipo 01, posições 1-based): DATA 3-10, CODBDI
11-12, CODNEG 13-24, TPMERC 25-27, PREULT 109-121 (2 casas), FATCOT 211-217.
"""
from __future__ import annotations

import io
import os
import zipfile

import pandas as pd

URL_COTAHIST = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"
CAMINHO_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "dados", "cotacoes_b3_mensais.csv.gz")
MERCADO_A_VISTA = "010"
CODBDI_FRACIONARIO = "96"


def ler_linhas_cotahist(linhas) -> pd.DataFrame:
    """Pregões do mercado à vista (lote padrão, FIIs, ETFs, units), de
    linhas do arquivo COTAHIST: data, ticker e fechamento em R$ por papel."""
    registros = []
    for linha in linhas:
        if isinstance(linha, bytes):
            linha = linha.decode("latin-1")
        if not linha.startswith("01") or len(linha) < 217:
            continue
        if linha[24:27] != MERCADO_A_VISTA or linha[10:12] == CODBDI_FRACIONARIO:
            continue
        try:
            fator = int(linha[210:217]) or 1
            fechamento = int(linha[108:121]) / 100 / fator
            data = pd.Timestamp(linha[2:10])
        except ValueError:
            continue
        registros.append((data, linha[12:24].strip(), fechamento))
    return pd.DataFrame(registros, columns=["data", "ticker", "fechamento"])


def ler_zip_cotahist(conteudo: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        nome = next(n for n in z.namelist() if n.upper().startswith("COTAHIST"))
        with z.open(nome) as f:
            return ler_linhas_cotahist(f)


_PADRAO_PAPEL = r"^[A-Z0-9]{4}(3|4|5|6|7|8|11)$"  # ações, units, FIIs e ETFs (fora BDRs, direitos, recibos)


def filtrar_papeis(pregoes: pd.DataFrame) -> pd.DataFrame:
    """Só ações (final 3 a 8) e papéis de final 11 (units, FIIs, ETFs) —
    deixa BDRs, direitos e recibos de fora, para a base ficar pequena."""
    if pregoes.empty:
        return pregoes
    return pregoes[pregoes["ticker"].str.match(_PADRAO_PAPEL)]


def resumo_mensal(pregoes: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por papel e mês: data do último pregão, fechamento nesse
    pregão e média dos fechamentos do mês."""
    if pregoes.empty:
        return pd.DataFrame(columns=["ticker", "mes", "data", "fechamento", "media", "pregoes"])
    df = pregoes.sort_values("data").copy()
    df["mes"] = df["data"].dt.to_period("M").astype(str)
    g = df.groupby(["ticker", "mes"])
    resumo = g.agg(data=("data", "last"), fechamento=("fechamento", "last"), media=("fechamento", "mean"),
                   pregoes=("fechamento", "size")).reset_index()
    resumo["fechamento"] = resumo["fechamento"].round(4)
    resumo["media"] = resumo["media"].round(4)
    return resumo


_CACHE_BASE: dict[str, pd.DataFrame] = {}


def carregar_base(caminho: str | None = None) -> pd.DataFrame | None:
    """Base mensal gerada pelo notebook; None se o arquivo não existir (o
    painel segue só com o Yahoo)."""
    caminho = caminho or CAMINHO_BASE
    if caminho not in _CACHE_BASE:
        if not os.path.exists(caminho):
            return None
        base = pd.read_csv(caminho, parse_dates=["data"])
        _CACHE_BASE[caminho] = base
    return _CACHE_BASE[caminho]


def _papel(ticker: str) -> str:
    return ticker.upper().replace(".SA", "")


def series_do_papel(base: pd.DataFrame | None, ticker: str) -> tuple[pd.Series, pd.Series]:
    """(fechamento no último pregão de cada mês, indexado pela data desse
    pregão; média dos fechamentos do mês, indexada pela mesma data)."""
    vazio = pd.Series(dtype=float)
    if base is None or base.empty:
        return vazio, vazio
    sub = base[base["ticker"] == _papel(ticker)].sort_values("data")
    if sub.empty:
        return vazio, vazio
    idx = pd.DatetimeIndex(sub["data"])
    return (pd.Series(sub["fechamento"].to_numpy(dtype=float), index=idx),
            pd.Series(sub["media"].to_numpy(dtype=float), index=idx))


def ajustar_desdobramentos(serie: pd.Series, eventos: pd.Series | None) -> pd.Series:
    """Põe preços sem ajuste (B3) na escala do Yahoo: cada desdobramento ou
    grupamento registrado pelo Yahoo (data -> fator, ex.: 10 num desdobramento
    de 1 para 10) divide os preços anteriores à data pelo fator."""
    if serie is None or serie.empty or eventos is None or len(eventos) == 0:
        return serie
    s = serie.astype(float).copy()
    ev = eventos.copy()
    if getattr(ev.index, "tz", None) is not None:
        ev.index = ev.index.tz_localize(None)
    for data, fator in ev.items():
        if fator and fator > 0 and fator != 1:
            s[s.index < pd.Timestamp(data)] = s[s.index < pd.Timestamp(data)] / float(fator)
    return s


def completar_precos(precos: pd.Series | None, base: pd.DataFrame | None, ticker: str,
                     eventos: pd.Series | None = None) -> tuple[pd.Series, pd.Timestamp | None]:
    """Acrescenta, ANTES do primeiro pregão do Yahoo, o fechamento mensal da
    B3 (um ponto por mês, no último pregão), na escala do Yahoo. Devolve
    (série completada, data do primeiro pregão do Yahoo) — a data é None
    quando nada foi acrescentado."""
    p = precos if precos is not None else pd.Series(dtype=float)
    if getattr(p.index, "tz", None) is not None:
        p = p.copy()
        p.index = p.index.tz_localize(None)
    fechamentos, _ = series_do_papel(base, ticker)
    if fechamentos.empty:
        return p, None
    inicio_yahoo = p.index[0] if len(p) else None
    antigos = fechamentos if inicio_yahoo is None else fechamentos[fechamentos.index.to_period("M")
                                                                   < inicio_yahoo.to_period("M")]
    if antigos.empty:
        return p, None
    antigos = ajustar_desdobramentos(antigos, eventos)
    return pd.concat([antigos, p]).sort_index(), inicio_yahoo


def medias_mensais(base: pd.DataFrame | None, ticker: str, eventos: pd.Series | None = None) -> pd.Series:
    """Média dos fechamentos de cada mês (B3), na escala do Yahoo — serve ao
    DY anual quando o Yahoo não cobre o ano inteiro."""
    _, medias = series_do_papel(base, ticker)
    return ajustar_desdobramentos(medias, eventos)
