"""Raio-X de um ativo: cálculos da aba (risco em 12 meses, pares do
segmento, proventos por ano e próximo provento). Sem I/O."""
from __future__ import annotations

import math

import pandas as pd

LIQUIDEZ_MINIMA_PAR = 1_000_000.0
MAXIMO_PARES = 8


def _sem_fuso(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype=float).dropna()
    idx = pd.DatetimeIndex(s.index)
    s.index = idx.tz_localize(None) if idx.tz is not None else idx
    return s.sort_index()


def risco_12m(precos: pd.Series | None) -> dict | None:
    """Maior queda (do topo ao fundo) e volatilidade anualizada nos últimos
    12 meses, a partir dos fechamentos diários."""
    if precos is None or len(precos) < 30:
        return None
    s = _sem_fuso(precos)
    s = s[s.index >= s.index[-1] - pd.DateOffset(months=12)]
    if len(s) < 30:
        return None
    queda = float((s / s.cummax() - 1).min())
    vol = float(s.pct_change().dropna().std() * math.sqrt(252))
    return {"maior_queda": queda, "volatilidade": vol}


def selecionar_pares(tabela: pd.DataFrame | None, papel: str, maximo: int = MAXIMO_PARES) -> pd.DataFrame | None:
    """Pares do segmento para a comparação: empresas com liquidez de pelo
    menos R$ 1 milhão/dia, as mais negociadas primeiro (com lucro ou não: o
    P/L negativo aparece como prejuízo). Outras classes da mesma empresa
    (PETR3 para PETR4) ficam de fora. O próprio ativo entra sempre; coluna
    ``proprio`` marca o ativo."""
    if tabela is None or tabela.empty:
        return None
    t = tabela.copy()
    papel = papel.upper().replace(".SA", "")
    t["proprio"] = t["papel"] == papel
    mesma_empresa = t["papel"].str[:4] == papel[:4]
    validos = t[(t["proprio"]) | (~mesma_empresa & (t["liquidez"].fillna(0) >= LIQUIDEZ_MINIMA_PAR))]
    validos = validos.sort_values("liquidez", ascending=False)
    escolhidos = validos.head(maximo)
    if not escolhidos["proprio"].any():
        escolhidos = pd.concat([escolhidos.head(maximo - 1), validos[validos["proprio"]]])
    return escolhidos.reset_index(drop=True) if len(escolhidos) > 1 else None


def posicao_entre_pares(pares: pd.DataFrame | None) -> str | None:
    """Frase curta: posição do ativo no P/L (menor primeiro) e no DY (maior
    primeiro) entre os pares."""
    if pares is None or not pares["proprio"].any():
        return None
    n = len(pares)
    proprio = pares[pares["proprio"]].iloc[0]
    partes = []
    if pd.notna(proprio.get("pl")) and proprio["pl"] > 0:
        pl = pares[pares["pl"] > 0]["pl"].rank(method="min")
        partes.append(f"{int(pl[pares['proprio']].iloc[0])}º menor P/L")
    if pd.notna(proprio.get("dy")):
        dy = pares["dy"].rank(method="min", ascending=False)
        partes.append(f"{int(dy[pares['proprio']].iloc[0])}º maior DY")
    return (" e ".join(partes) + f" entre {n}") if partes else None


_CHAVE_TIPO = {"jcp": "jcp", "rendimento": "rendimentos", "amortizacao": "amortizacao"}


def proventos_por_ano(detalhe: pd.DataFrame | None, anos: list[int]) -> dict[int, dict]:
    """Por ano da data-com, por ação/cota (valores da época): dividendos e JCP
    (ações) ou rendimentos e amortizações (FIIs)."""
    saida = {a: {"dividendos": 0.0, "jcp": 0.0, "rendimentos": 0.0, "amortizacao": 0.0} for a in anos}
    if detalhe is None or detalhe.empty:
        return saida
    for _, p in detalhe.iterrows():
        a = int(pd.Timestamp(p["data_com"]).year)
        if a in saida:
            saida[a][_CHAVE_TIPO.get(p["tipo"], "dividendos")] += float(p["valor"])
    return saida


def proximo_provento(detalhe: pd.DataFrame | None, hoje) -> dict | None:
    """O provento anunciado mais próximo de ser pago: pagamento depois de hoje
    ou ainda sem data com data-com recente (últimos 13 meses)."""
    if detalhe is None or detalhe.empty:
        return None
    hoje = pd.Timestamp(hoje)
    pag = pd.to_datetime(detalhe["pagamento"])
    recente = pd.to_datetime(detalhe["data_com"]) >= hoje - pd.Timedelta(days=400)
    futuros = detalhe[(pag > hoje) | (pag.isna() & recente)]
    if futuros.empty:
        return None
    futuros = futuros.assign(_ord=pd.to_datetime(futuros["pagamento"]).fillna(pd.Timestamp.max))
    p = futuros.sort_values(["_ord", "data_com"]).iloc[0]
    return {"tipo": p["tipo"], "valor": float(p["valor"]), "data_com": pd.Timestamp(p["data_com"]),
            "pagamento": None if pd.isna(p["pagamento"]) else pd.Timestamp(p["pagamento"])}


def pares_fii(todos: dict | None, ticker: str, maximo: int = MAXIMO_PARES) -> pd.DataFrame | None:
    """FIIs do mesmo segmento (Fundamentus), mais negociados primeiro; o
    próprio fundo entra sempre."""
    if not todos:
        return None
    papel = ticker.upper().replace(".SA", "")
    proprio = todos.get(papel)
    if proprio is None or not getattr(proprio, "segmento", None):
        return None
    linhas = [{"papel": p, "dy": f.dividend_yield, "pvp": f.pvp, "liquidez": f.liquidez, "proprio": p == papel}
              for p, f in todos.items() if getattr(f, "segmento", None) == proprio.segmento]
    t = pd.DataFrame(linhas)
    t = t[(t["proprio"]) | ((t["liquidez"].fillna(0) >= LIQUIDEZ_MINIMA_PAR) & (t["dy"].fillna(0) > 0))]
    t = t.sort_values("liquidez", ascending=False)
    escolhidos = t.head(maximo)
    if not escolhidos["proprio"].any():
        escolhidos = pd.concat([escolhidos.head(maximo - 1), t[t["proprio"]]])
    return escolhidos.reset_index(drop=True) if len(escolhidos) > 1 else None
