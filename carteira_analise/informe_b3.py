"""Conferência dos proventos do painel com o relatório consolidado anual da B3
(Área do Investidor > Relatórios > Consolidado anual, aba "Proventos
Recebidos": Produto, Tipo de Evento, Valor líquido).

Não faz I/O: recebe as abas já lidas (``pd.read_excel(..., sheet_name=None)``)
e a tabela do ano do painel (``proventos_ir.resumo_por_ano``). O informe fica
só na sessão do navegador, como a planilha.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

TOLERANCIA_REAIS = 1.0     # diferença até R$ 1 ...
TOLERANCIA_RELATIVA = 0.01  # ... ou até 1% do valor conta como "bate"

# tipo de evento da B3 -> categoria do painel
_CATEGORIAS = {
    "dividendo": "dividendos",
    "juros sobre capital proprio": "jcp",
    "rendimento": "rendimentos",
    "amortizacao": "amortizacao",
    "restituicao de capital": "restituicao",
}


def _sem_acento(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t).strip().lower()


def _numero(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).replace("R$", "").strip()
    if not t or t == "-":
        return None
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def ticker_base(produto: str) -> tuple[str, bool]:
    """Código de negociação sem o recibo de subscrição: "CPTS11B" -> ("CPTS11", True)."""
    p = str(produto).strip().upper().split(" ")[0]
    if re.fullmatch(r"[A-Z]{4}\d{1,2}B", p):
        return p[:-1], True
    return p, False


def ano_do_nome(nome_arquivo: str | None) -> int | None:
    """Ano no nome do arquivo da B3 ("relatorio-consolidado-anual-2025.xlsx")."""
    m = re.search(r"(20\d{2})", str(nome_arquivo or ""))
    return int(m.group(1)) if m else None


def ler_proventos_informe(abas: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Aba de proventos do informe -> DataFrame (produto, ticker, recibo,
    categoria, evento, valor). None se o arquivo não tiver essa aba."""
    aba = next((df for nome, df in abas.items() if "provento" in _sem_acento(nome)), None)
    if aba is None or aba.empty:
        return None
    colunas = {_sem_acento(c): c for c in aba.columns}
    c_prod = next((colunas[k] for k in colunas if k.startswith("produto")), None)
    c_tipo = next((colunas[k] for k in colunas if "tipo" in k and "evento" in k), None)
    c_valor = next((colunas[k] for k in colunas if "valor liquido" in k), None)
    if not (c_prod and c_tipo and c_valor):
        return None
    linhas = []
    for _, r in aba.iterrows():
        produto, evento, valor = r[c_prod], r[c_tipo], _numero(r[c_valor])
        if pd.isna(produto) or pd.isna(evento) or valor is None:
            continue  # linhas de total e em branco
        ticker, recibo = ticker_base(produto)
        linhas.append({"produto": str(produto).strip(), "ticker": ticker, "recibo": recibo,
                       "categoria": _CATEGORIAS.get(_sem_acento(evento), "outro"),
                       "evento": str(evento).strip(), "valor": valor})
    return pd.DataFrame(linhas, columns=["produto", "ticker", "recibo", "categoria", "evento", "valor"])


def _bate(painel: float, b3: float) -> bool:
    return abs(painel - b3) <= max(TOLERANCIA_REAIS, TOLERANCIA_RELATIVA * max(abs(b3), abs(painel)))


def conciliar(informe: pd.DataFrame, tabela_painel: pd.DataFrame, tickers_planilha: set[str],
              fiis: set[str]) -> pd.DataFrame:
    """Uma linha por ativo e categoria (dividendos, JCP líquido, rendimentos,
    amortização) com o valor do informe, o do painel, a diferença e o
    diagnóstico. ``tabela_painel``: ``resumo_por_ano`` (tickers com ou sem
    .SA); ``tickers_planilha`` e ``fiis``: códigos sem .SA."""
    painel = {}
    if tabela_painel is not None and len(tabela_painel):
        for _, r in tabela_painel.iterrows():
            t = str(r["ticker"]).replace(".SA", "")
            painel[t] = {"dividendos": float(r["dividendos"]), "jcp": float(r["jcp_liquido"]),
                         "rendimentos": float(r["rendimentos"]), "amortizacao": float(r["amortizacao"])}
    b3: dict[str, dict[str, float]] = {}
    recibos: dict[str, list[str]] = {}
    extras: list[dict] = []
    for r in informe.itertuples():
        t = r.ticker
        if r.recibo:
            recibos.setdefault(t, []).append(r.produto)
        if r.categoria == "rendimentos" and t not in fiis and not re.fullmatch(r"[A-Z]{4}11", t):
            extras.append({"ticker": t, "categoria": "atualizacao", "b3": r.valor})  # ação: atualização monetária
            continue
        if r.categoria in ("restituicao", "outro"):
            extras.append({"ticker": t, "categoria": r.categoria, "b3": r.valor, "evento": r.evento})
            continue
        b3.setdefault(t, {}).setdefault(r.categoria, 0.0)
        b3[t][r.categoria] += r.valor
    linhas = []
    for t in sorted(set(b3) | set(painel)):
        for cat in ("dividendos", "jcp", "rendimentos", "amortizacao"):
            v_b3 = b3.get(t, {}).get(cat, 0.0)
            v_p = painel.get(t, {}).get(cat, 0.0)
            if abs(v_b3) < 0.005 and abs(v_p) < 0.005:
                continue
            if t not in tickers_planilha:
                diag, sit = "não está na planilha", "fora"
            elif t not in b3:
                diag, sit = "não aparece no informe: outra corretora ou conta?", "diferente"
            elif _bate(v_p, v_b3):
                diag, sit = "bate", "ok"
            elif v_p < v_b3:
                diag, sit = "faltou no painel: compras antigas não lançadas ou quantidade na data-com", "diferente"
            else:
                diag, sit = "a mais no painel: vendas não lançadas ou quantidade na data-com", "diferente"
            if t in recibos and cat == "rendimentos":
                diag += f" (informe inclui o recibo {', '.join(sorted(set(recibos[t])))})"
            linhas.append({"ticker": t, "categoria": cat, "b3": v_b3, "painel": v_p, "diferenca": v_p - v_b3,
                           "situacao": sit, "diagnostico": diag})
    for e in extras:
        diag = {"atualizacao": "atualização de provento pago com atraso (fora do Fundamentus)",
                "restituicao": "não é renda: reduz o custo da ação"}.get(
            e["categoria"], f"{e.get('evento', 'outro evento')}: confira no informe")
        linhas.append({"ticker": e["ticker"], "categoria": e["categoria"], "b3": e["b3"], "painel": 0.0,
                       "diferenca": -e["b3"], "situacao": "informativo", "diagnostico": diag})
    return pd.DataFrame(linhas, columns=["ticker", "categoria", "b3", "painel", "diferenca", "situacao",
                                         "diagnostico"])


def resumo_conciliacao(c: pd.DataFrame) -> dict:
    """Totais por categoria (informe × painel) e contagem por situação."""
    tot = {}
    for cat in ("dividendos", "jcp", "rendimentos", "amortizacao"):
        parte = c[c["categoria"] == cat]
        if len(parte):
            na = parte[parte["situacao"] != "fora"]
            tot[cat] = {"b3": float(parte["b3"].sum()), "painel": float(parte["painel"].sum()),
                        "b3_planilha": float(na["b3"].sum())}
    return {"totais": tot, "contagem": c["situacao"].value_counts().to_dict()}
