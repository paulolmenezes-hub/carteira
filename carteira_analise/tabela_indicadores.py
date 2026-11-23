"""Tabela de indicadores da carteira (aba Dashboard): uma linha por posição
aberta, com os números que já aparecem nos cards, para comparar os ativos
lado a lado. Separada por grupo: Ações B3, FIIs, ETFs B3 e EUA. Os valores
ficam numéricos (a tabela ordena por coluna); o texto vem só onde é leitura."""
from __future__ import annotations

import re

import pandas as pd

FAIXA = {"realizacao": "📈 Realização", "queda": "📉 Queda", "fora": "↔️ Fora"}
SELO = {"regular": "Regular", "oscilante": "Oscilante", "irregular": "Irregular", "incompleto": "Incompleto"}
PRECO_ATRATIVO = {"forte": "Sim, forte", "sim": "Sim", "desconto": "Com desconto", "nao": "Não"}
_NUMERICAS = {"P/L", "P/VP", "P/VP médio 24m", "ROE (%)", "ROIC (%)", "EV/EBITDA", "DY 12m (%)", "Preço",
              "Preço-teto", "Preço × teto (%)", "Variação s/ PM (%)"}
_MES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def _numero(texto) -> float | None:
    """'6,1' -> 6.1; '11,5%' -> 11.5; '—' -> None."""
    if texto is None:
        return None
    m = re.search(r"-?[\d.]*\d(?:,\d+)?", str(texto).replace("−", "-"))
    if not m:
        return None
    try:
        return float(m.group(0).replace(".", "").replace(",", "."))
    except ValueError:
        return None


def _indicador(ctx: dict | None, prefixo: str) -> dict | None:
    for ind in (ctx or {}).get("indicadores") or []:
        if str(ind.get("nome", "")).startswith(prefixo):
            return ind
    return None


def meses_de_pagamento(dividendos: pd.Series | None, hoje) -> str:
    """Meses com data-ex nos últimos 24 meses ("Fev · Ago"); "Mensal" quando
    paga em quase todos os meses."""
    if dividendos is None or len(dividendos) == 0:
        return "—"
    idx = pd.DatetimeIndex(dividendos.index)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    recentes = idx[idx >= pd.Timestamp(hoje) - pd.DateOffset(months=24)]
    if len(recentes) == 0:
        return "—"
    ultimos12 = idx[idx >= pd.Timestamp(hoje) - pd.DateOffset(months=12)]
    if len(set(ultimos12.month)) >= 10:
        return "Mensal"
    return " · ".join(_MES[m - 1] for m in sorted(set(recentes.month)))


def grupo_da_tabela(card: dict) -> str:
    if card.get("mercado") == "EUA":
        return "EUA"
    return {"Ações": "Ações B3", "FIIs": "FIIs", "ETF": "ETFs B3"}.get(card.get("grupo"), "Ações B3")


def montar_tabelas(cards: list[dict], fundamentos: dict[str, dict], dividendos: dict[str, pd.Series],
                   hoje) -> dict[str, pd.DataFrame]:
    """{grupo: DataFrame} com as colunas de cada grupo; índice = ticker."""
    grupos: dict[str, list[dict]] = {}
    for c in cards:
        t = c["ticker"]
        ctx = c.get("contexto") or {}
        fund = fundamentos.get(t) or {}
        dy = _indicador(ctx, "Dividend yield") or _indicador(ctx, "Yield")
        base = {"Ativo": t.replace(".SA", ""), "Faixa": FAIXA.get(c.get("situacao"), "—"),
                "Preço": c.get("preco_atual"),
                "Variação s/ PM (%)": (c.get("variacao") * 100) if c.get("variacao") is not None else None,
                "DY 12m (%)": _numero(dy.get("valor")) if dy else None,
                "Meses (data-com)": meses_de_pagamento(dividendos.get(t), hoje)}
        g = grupo_da_tabela(c)
        if g == "Ações B3":
            pl = _indicador(ctx, "P/L")
            pt = ctx.get("preco_teto") or {}
            teto = pt.get("teto")
            reg = (ctx.get("regularidade") or {}).get("selo")
            leitura_pl = (pl or {}).get("leitura") or ""
            linha = {**base,
                     "P/L": _numero(pl.get("valor")) if pl else fund.get("pl"),
                     "P/L × referência": leitura_pl.split(" · lucro")[0] if leitura_pl else "—",
                     "P/VP": fund.get("pvp"),
                     "ROE (%)": fund.get("roe") * 100 if fund.get("roe") is not None else None,
                     "ROIC (%)": fund.get("roic") * 100 if fund.get("roic") else None,
                     # 0 no Fundamentus = não se aplica (bancos e seguradoras não têm EBITDA/ROIC)
                     "EV/EBITDA": fund.get("ev_ebitda") or None,
                     "Preço-teto": teto,
                     "Preço × teto (%)": ((c.get("preco_atual") / teto - 1) * 100) if teto and c.get("preco_atual") else None,
                     "Dividendos": SELO.get(reg, "—")}
            ordem = ["Ativo", "Faixa", "P/L", "P/L × referência", "P/VP", "ROE (%)", "ROIC (%)", "EV/EBITDA",
                     "DY 12m (%)", "Preço", "Preço-teto", "Preço × teto (%)", "Dividendos", "Meses (data-com)"]
        elif g == "FIIs":
            pa = ctx.get("preco_atrativo") or {}
            linha = {**base, "Segmento": fund.get("segmento") or "—",
                     "P/VP": pa.get("pvp") if pa.get("pvp") is not None else fund.get("pvp"),
                     "P/VP médio 24m": pa.get("pvp_medio_24m"),
                     "Preço atrativo?": PRECO_ATRATIVO.get(pa.get("nivel"), "—")}
            ordem = ["Ativo", "Faixa", "Segmento", "DY 12m (%)", "P/VP", "P/VP médio 24m", "Preço atrativo?",
                     "Preço", "Variação s/ PM (%)", "Meses (data-com)"]
        else:
            tend = _indicador(ctx, "Tendência")
            linha = {**base, "Tipo": "Ação" if c.get("grupo") == "Ações" else "ETF",
                     "Tendência (MM200)": (tend or {}).get("valor") or "—"}
            ordem = ["Ativo", "Tipo", "Faixa", "Preço", "Variação s/ PM (%)", "DY 12m (%)", "Tendência (MM200)",
                     "Meses (data-com)"]
        grupos.setdefault(g, []).append({k: linha.get(k) for k in ordem} | {"_ticker": t})
    saida = {}
    for g in ("Ações B3", "FIIs", "ETFs B3", "EUA"):
        if g in grupos:
            df = pd.DataFrame(grupos[g])
            for col in _NUMERICAS & set(df.columns):  # vazio vira NaN (célula em branco), não "None"
                df[col] = pd.to_numeric(df[col], errors="coerce")
            saida[g] = df.set_index("_ticker")
    return saida
