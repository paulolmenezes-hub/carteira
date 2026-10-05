"""Testes de "preço atrativo" para FIIs (Sprint 4).

Em vez de prever o movimento do preço (o sinal técnico ficou próximo de
"não fazer nada"), estes critérios perguntam se o FII está BARATO em
relação a ele mesmo ou à renda fixa — e o backtest mede se, no passado,
isso aumentou a chance de o retorno dos 12 meses seguintes superar o CDI.

Critérios (fixados ANTES de ver os resultados, para não ajustar ao passado):

- Caminho 2 — yield vs. o próprio histórico: yield dos últimos 12 meses
  ao menos 15% acima da média do próprio FII nos 24 meses anteriores, com
  rendimentos estáveis: queda de no máximo 10% sobre os 12 meses anteriores
  E nos últimos 3 meses (anualizados) — a soma de 12 meses demora a mostrar
  um corte recente, e o yield "alto" de um fundo em deterioração seria
  confundido com oportunidade (a armadilha do yield).
- Caminho 3 — prêmio sobre o Tesouro IPCA+: prêmio (yield − juro real do
  Tesouro IPCA+) ao menos 1 ponto percentual acima da média do próprio
  prêmio nos 24 meses anteriores, com rendimentos estáveis.

Resultado de cada caso (FII, mês): o retorno total dos 12 meses seguintes
(preço + rendimentos recebidos, sem reinvestir) superou o CDI do período,
líquido de IR (17,5% — aplicação de 12 meses)? Compara a taxa de acerto
nos casos COM o critério com a taxa em TODOS os casos (taxa-base). Como os
casos de um mesmo FII se sobrepõem no tempo, o intervalo de confiança da
diferença usa reamostragem por FII (bootstrap por grupo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DESCONTO_MINIMO_YIELD = 0.15     # caminho 2: yield 15% acima da própria média
PREMIO_EXTRA_MINIMO = 0.01       # caminho 3: prêmio 1 p.p. acima da própria média
QUEDA_MAXIMA_RENDIMENTOS = 0.10  # rendimentos "estáveis": queda de no máximo 10%
MESES_MEDIA = 24
IR_CDI_12M = 0.175               # IR de aplicação entre 361 e 720 dias


def _sem_fuso(serie):
    if serie is None:
        return pd.Series(dtype=float)
    if getattr(serie.index, "tz", None) is not None:
        serie = serie.copy()
        serie.index = serie.index.tz_localize(None)
    return serie.sort_index()


def avaliar_fii_mes_a_mes(precos: pd.Series, dividendos: pd.Series, indice_cdi: pd.Series,
                          juro_real: pd.Series | None = None) -> pd.DataFrame:
    """Uma linha por fim de mês avaliável de um FII: yield, média do próprio
    yield, prêmio sobre o Tesouro IPCA+, critérios (caminhos 2 e 3), retorno
    dos 12 meses seguintes e CDI líquido do mesmo período."""
    p = _sem_fuso(precos).dropna()
    d = _sem_fuso(dividendos)
    if len(p) < 60:
        return pd.DataFrame()
    cdi = indice_cdi.reindex(p.index).ffill() if indice_cdi is not None else None
    juro = None
    if juro_real is not None and len(juro_real):
        juro = _sem_fuso(juro_real).reindex(_sem_fuso(juro_real).index.union(p.index)).ffill().reindex(p.index)

    def soma_div(ini, fim):
        return float(d[(d.index > ini) & (d.index <= fim)].sum()) if len(d) else 0.0

    # O yield só é calculado com 12 meses COMPLETOS de histórico (de preço e de
    # rendimentos): no começo da série a soma de 12 meses ainda está incompleta,
    # o yield parece baixo e puxaria a média para baixo, disparando o critério
    # sem motivo (ex.: FII recém-listado).
    inicio_dados = max(p.index[0], d.index[0]) if len(d) else p.index[0]
    # Janelas em MESES do calendário (e não em dias): com rendimentos mensais, uma
    # janela de 91 dias pegaria 2 ou 3 pagamentos conforme o dia do mês.
    m = pd.DateOffset
    fins_de_mes = p.groupby([p.index.year, p.index.month]).tail(1).index
    linhas = []
    for t in fins_de_mes:
        if t < inicio_dados + m(months=12):
            continue
        preco = float(p.loc[t])
        div12 = soma_div(t - m(months=12), t)
        if preco <= 0 or div12 <= 0:
            continue
        tem_ano_anterior = t >= inicio_dados + m(months=24)
        # últimos 3 meses (anualizados) x os 12 meses que terminam 3 meses antes:
        # pega cortes recentes que a soma de 12 meses ainda esconde (yield trap)
        recentes_anualizado = soma_div(t - m(months=3), t) * 4
        referencia_recentes = (soma_div(t - m(months=15), t - m(months=3))
                               if t >= inicio_dados + m(months=15) else np.nan)
        linhas.append({"data": t, "preco": preco, "div12": div12, "yield": div12 / preco,
                       "div12_anterior": (soma_div(t - m(months=24), t - m(months=12))
                                          if tem_ano_anterior else np.nan),
                       "div3m_anualizado": recentes_anualizado, "div12_ate_3m_atras": referencia_recentes,
                       "juro_real": float(juro.loc[t]) if juro is not None and pd.notna(juro.loc[t]) else None})
    df = pd.DataFrame(linhas)
    if df.empty:
        return df
    df["premio"] = df["yield"] - df["juro_real"].astype(float)
    df["yield_medio_24m"] = df["yield"].shift(1).rolling(MESES_MEDIA, min_periods=MESES_MEDIA).mean()
    df["premio_medio_24m"] = df["premio"].shift(1).rolling(MESES_MEDIA, min_periods=MESES_MEDIA).mean()
    df["estavel"] = ((df["div12_anterior"] > 0)
                     & (df["div12"] >= (1 - QUEDA_MAXIMA_RENDIMENTOS) * df["div12_anterior"])
                     & (df["div3m_anualizado"] >= (1 - QUEDA_MAXIMA_RENDIMENTOS) * df["div12_ate_3m_atras"]))
    sem_comparacao = df["div12_anterior"].isna()
    # critérios como 1/0, e vazio (NaN) enquanto não há 24 meses de histórico para a média
    crit_y = (df["yield"] >= (1 + DESCONTO_MINIMO_YIELD) * df["yield_medio_24m"]) & df["estavel"]
    crit_p = (df["premio"] >= df["premio_medio_24m"] + PREMIO_EXTRA_MINIMO) & df["estavel"]
    df["criterio_yield"] = np.where(df["yield_medio_24m"].isna() | sem_comparacao, np.nan, crit_y.astype(float))
    df["criterio_premio"] = np.where(df["premio_medio_24m"].isna() | sem_comparacao, np.nan, crit_p.astype(float))

    # resultado: retorno total em 12 meses x CDI líquido do mesmo período
    ultimo = p.index[-1]
    ret, cdi_liq = [], []
    for t, preco in zip(df["data"], df["preco"]):
        fim = t + pd.DateOffset(months=12)
        if fim > ultimo:
            ret.append(np.nan)
            cdi_liq.append(np.nan)
            continue
        preco_fim = float(p[p.index <= fim].iloc[-1])
        ret.append((preco_fim + soma_div(t, fim)) / preco - 1)
        if cdi is not None:
            i0, i1 = float(cdi.loc[t]), float(cdi[cdi.index <= fim].iloc[-1])
            cdi_liq.append((i1 / i0 - 1) * (1 - IR_CDI_12M))
        else:
            cdi_liq.append(0.0)
    df["retorno_12m"] = ret
    df["cdi_liquido_12m"] = cdi_liq
    df["superou_cdi"] = np.where(df["retorno_12m"].isna(), np.nan,
                                 (df["retorno_12m"] > df["cdi_liquido_12m"]).astype(float))
    return df


def _taxa(df: pd.DataFrame) -> float:
    return float(df["superou_cdi"].astype(float).mean()) if len(df) else float("nan")


def resumir_criterio(casos: pd.DataFrame, coluna: str, n_bootstrap: int = 2000,
                     semente: int = 42) -> dict | None:
    """Compara a taxa de acerto (superou o CDI líquido em 12 meses) nos casos
    com o critério contra a taxa-base (todos os casos avaliáveis), com
    intervalo de confiança de 90% da diferença por bootstrap por FII."""
    base = casos[casos[coluna].notna() & casos["superou_cdi"].notna()].copy()
    if base.empty:
        return None
    base[coluna] = base[coluna].astype(float) > 0.5
    com = base[base[coluna]]
    resumo = {
        "fiis": int(base["ticker"].nunique()), "casos": int(len(base)),
        "casos_com_criterio": int(len(com)), "fiis_com_criterio": int(com["ticker"].nunique()),
        "taxa_base": _taxa(base), "taxa_com_criterio": _taxa(com) if len(com) else None,
        "excesso_mediano_base": float((base["retorno_12m"] - base["cdi_liquido_12m"]).median()),
        "excesso_mediano_com_criterio": (float((com["retorno_12m"] - com["cdi_liquido_12m"]).median())
                                         if len(com) else None),
        "periodo": (base["data"].min(), base["data"].max()),
        "ic90_diferenca": None,
    }
    if len(com) and n_bootstrap:
        rng = np.random.default_rng(semente)
        grupos = {t: g for t, g in base.groupby("ticker")}
        tickers = list(grupos)
        difs = []
        for _ in range(n_bootstrap):
            amostra = pd.concat([grupos[t] for t in rng.choice(tickers, size=len(tickers), replace=True)])
            c = amostra[amostra[coluna]]
            if len(c):
                difs.append(_taxa(c) - _taxa(amostra))
        if difs:
            resumo["ic90_diferenca"] = (float(np.percentile(difs, 5)), float(np.percentile(difs, 95)))
    return resumo


def frase_resultado(resumo: dict | None, descricao_criterio: str) -> str:
    """Leitura em linguagem simples do resultado de um critério."""
    if not resumo:
        return f"{descricao_criterio}: sem casos avaliáveis."
    if not resumo["casos_com_criterio"]:
        return f"{descricao_criterio}: o critério não ocorreu no período testado."
    ini, fim = resumo["periodo"]
    dif = resumo["taxa_com_criterio"] - resumo["taxa_base"]
    frase = (f"Em {resumo['fiis']} FIIs, de {ini:%m/%Y} a {fim:%m/%Y}: {descricao_criterio}, o retorno dos 12 meses "
             f"seguintes superou o CDI líquido em {resumo['taxa_com_criterio'] * 100:.0f}% dos "
             f"{resumo['casos_com_criterio']} casos, contra {resumo['taxa_base'] * 100:.0f}% em todos os "
             f"{resumo['casos']} casos (diferença de {dif * 100:+.0f} pontos percentuais).")
    ic = resumo.get("ic90_diferenca")
    if ic:
        frase += f" Intervalo de confiança de 90% da diferença: {ic[0] * 100:+.0f} a {ic[1] * 100:+.0f} p.p."
        if ic[0] > 0:
            frase += " 👉 O critério aumentou a chance de superar o CDI, com margem estatística."
        elif ic[1] < 0:
            frase += " 👉 O critério DIMINUIU a chance de superar o CDI."
        else:
            frase += " 👉 Sem diferença confiável: o resultado pode ser acaso."
    return frase
