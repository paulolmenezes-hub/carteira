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
    líquido, rendimentos de FII (isentos), amortizações de FII (devolução de
    capital: reduzem o custo, não são rendimento) e total líquido recebido.
    Proventos sem data de pagamento ficam de fora (ver ``a_receber``)."""
    colunas = ["ticker", "dividendos", "jcp_bruto", "ir_retido", "jcp_liquido", "rendimentos", "amortizacao",
               "total_liquido"]
    if recebidos is None or recebidos.empty:
        return pd.DataFrame(columns=colunas)
    r = recebidos[recebidos["pagamento"].notna() & (pd.to_datetime(recebidos["pagamento"]).dt.year == ano)]
    linhas = []
    for t, g in r.groupby("ticker"):
        soma = lambda *tipos: float(g.loc[g["tipo"].isin(tipos), "bruto"].sum())
        div, rend, amort = soma("dividendo", "outro"), soma("rendimento"), soma("amortizacao")
        jcp = g[g["tipo"] == "jcp"]
        jb, ir = float(jcp["bruto"].sum()), float(jcp["ir_retido"].sum())
        linhas.append({"ticker": t, "dividendos": div, "jcp_bruto": jb, "ir_retido": ir, "jcp_liquido": jb - ir,
                       "rendimentos": rend, "amortizacao": amort, "total_liquido": div + jb - ir + rend + amort})
    return pd.DataFrame(linhas, columns=colunas).sort_values("ticker").reset_index(drop=True)


JANELA_A_RECEBER_DIAS = 400  # sem data de pagamento há mais tempo que isso: provavelmente já pago


def a_receber(recebidos: pd.DataFrame, hoje) -> pd.DataFrame:
    """Proventos com data-com passada e pagamento futuro, ou ainda sem data
    de pagamento com data-com recente (até ``JANELA_A_RECEBER_DIAS``). Sem
    data há mais tempo, o Fundamentus só não registrou a data (ver
    ``sem_data_antigos``)."""
    if recebidos is None or recebidos.empty:
        return recebidos
    pag = pd.to_datetime(recebidos["pagamento"])
    recente = pd.to_datetime(recebidos["data_com"]) >= pd.Timestamp(hoje) - pd.Timedelta(days=JANELA_A_RECEBER_DIAS)
    return (recebidos[(pag > pd.Timestamp(hoje)) | (pag.isna() & recente)]
            .sort_values("data_com").reset_index(drop=True))


def sem_data_antigos(recebidos: pd.DataFrame, hoje) -> pd.DataFrame:
    """Proventos antigos sem data de pagamento no Fundamentus: não entram em
    nenhum ano da tabela; conferir no informe."""
    if recebidos is None or recebidos.empty:
        return recebidos
    pag = pd.to_datetime(recebidos["pagamento"])
    antigo = pd.to_datetime(recebidos["data_com"]) < pd.Timestamp(hoje) - pd.Timedelta(days=JANELA_A_RECEBER_DIAS)
    return recebidos[pag.isna() & antigo].sort_values("data_com").reset_index(drop=True)


def eh_bdr(ticker: str) -> bool:
    """BDRs (final 32, 33, 34, 35 ou 39) seguem a regra de rendimentos do
    exterior, não a de dividendos de empresa brasileira."""
    codigo = ticker.upper().replace(".SA", "")
    return len(codigo) >= 6 and codigo[-2:] in ("32", "33", "34", "35", "39")
