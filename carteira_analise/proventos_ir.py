"""Proventos para a declaração do IR (ações da B3).

Para cada provento registrado no Fundamentus (tipo e data de pagamento),
calcula o que a planilha recebeu: quantidade na data-com (escala da época,
como o valor por ação do Fundamentus) x valor por ação. Separa dividendos
(isentos) e JCP (15% de IR retido na fonte; declara-se o líquido) e agrupa
pelo ANO DO PAGAMENTO, que é o que vale na declaração. Serve para conferir e
organizar: o documento oficial é o informe de rendimentos da corretora ou
da empresa.
"""
from __future__ import annotations

import pandas as pd

ALIQUOTA_JCP = 0.15


def quantidade_na_data_com(ops: list[dict], data_com) -> float:
    """Ações ao fim do dia da data-com (operações do próprio dia contam:
    quem compra na data-com recebe)."""
    q = 0.0
    for op in sorted(ops, key=lambda o: pd.Timestamp(o["data"])):
        if pd.Timestamp(op["data"]) > pd.Timestamp(data_com):
            break
        if op["tipo"] in ("compra", "bonificacao", "desdobramento"):
            q += float(op["quantidade"])
        elif op["tipo"] in ("venda", "grupamento"):
            q -= float(op["quantidade"])
    return max(q, 0.0)


def proventos_recebidos(ticker: str, ops: list[dict], detalhe: pd.DataFrame | None) -> pd.DataFrame:
    """Um provento recebido por linha: ticker, data_com, pagamento, tipo,
    quantidade, bruto, ir_retido, liquido."""
    colunas = ["ticker", "data_com", "pagamento", "tipo", "quantidade", "bruto", "ir_retido", "liquido"]
    if detalhe is None or detalhe.empty or not ops:
        return pd.DataFrame(columns=colunas)
    inicio = min(pd.Timestamp(o["data"]) for o in ops)
    linhas = []
    for _, p in detalhe.iterrows():
        if p["data_com"] < inicio:
            continue
        q = quantidade_na_data_com(ops, p["data_com"])
        if q <= 1e-9:
            continue
        bruto = q * float(p["valor"])
        ir = bruto * ALIQUOTA_JCP if p["tipo"] == "jcp" else 0.0
        linhas.append({"ticker": ticker, "data_com": p["data_com"], "pagamento": p["pagamento"], "tipo": p["tipo"],
                       "quantidade": q, "bruto": bruto, "ir_retido": ir, "liquido": bruto - ir})
    return pd.DataFrame(linhas, columns=colunas)


def resumo_por_ano(recebidos: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Por ativo, no ano de PAGAMENTO: dividendos, JCP bruto, IR retido, JCP
    líquido e total líquido. Proventos sem data de pagamento ficam de fora
    (ver ``a_receber``)."""
    colunas = ["ticker", "dividendos", "jcp_bruto", "ir_retido", "jcp_liquido", "total_liquido"]
    if recebidos is None or recebidos.empty:
        return pd.DataFrame(columns=colunas)
    r = recebidos[recebidos["pagamento"].notna() & (pd.to_datetime(recebidos["pagamento"]).dt.year == ano)]
    linhas = []
    for t, g in r.groupby("ticker"):
        div = float(g.loc[g["tipo"] != "jcp", "bruto"].sum())
        jcp = g[g["tipo"] == "jcp"]
        jb, ir = float(jcp["bruto"].sum()), float(jcp["ir_retido"].sum())
        linhas.append({"ticker": t, "dividendos": div, "jcp_bruto": jb, "ir_retido": ir,
                       "jcp_liquido": jb - ir, "total_liquido": div + jb - ir})
    return pd.DataFrame(linhas, columns=colunas).sort_values("ticker").reset_index(drop=True)


def a_receber(recebidos: pd.DataFrame, hoje) -> pd.DataFrame:
    """Proventos com data-com passada e pagamento futuro ou ainda sem data."""
    if recebidos is None or recebidos.empty:
        return recebidos
    pag = pd.to_datetime(recebidos["pagamento"])
    return recebidos[pag.isna() | (pag > pd.Timestamp(hoje))].reset_index(drop=True)
