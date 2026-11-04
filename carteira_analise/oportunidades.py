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

- Caminho 4 — P/VP (incluído em 03/11/2026, com critérios fixados antes de
  rodar; o valor patrimonial por cota vem dos informes mensais da CVM e
  entra com 1 mês de defasagem, o prazo de publicação do informe):
  4a: P/VP abaixo de 0,95, com rendimentos estáveis;
  4b: P/VP 10% ou mais abaixo da média do próprio FII nos 24 meses
      anteriores, com rendimentos estáveis;
  4c: caminho 2 E P/VP abaixo de 0,95 (o yield alto confirmado pelo desconto).

- Etapa C — renda estável descontada a inflação (incluída em 04/11/2026,
  critérios fixados antes de rodar), só para FIIs de papel: os caminhos 2,
  4a e 4c ganham variantes "_real" que aceitam rendimentos estáveis OU
  estáveis acima da inflação (ver ``renda_acima_da_inflacao``). Motivo:
  em fundos de CRI atrelados ao IPCA, a queda do rendimento nominal quando
  a inflação desacelera não é deterioração do fundo (caso KNIP11).

Lacunas na fonte: o histórico de rendimentos de FIIs no Yahoo Finance tem
meses sem registro. Só janelas de 12 meses com ao menos 10 pagamentos contam
(no histórico dos critérios e nos 12 meses seguintes), e a média de 24 meses
exige 24 meses seguidos válidos — uma lacuna invalida a média em vez de
distorcê-la. FIIs com pagamentos não mensais ficam de fora.

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
MIN_PAGAMENTOS_12M = 10          # janela de 12 meses "completa": ao menos 10 pagamentos mensais
MESES_MEDIA = 24
IR_CDI_12M = 0.175               # IR de aplicação entre 361 e 720 dias
PVP_DESCONTO = 0.95              # caminho 4a/4c: P/VP abaixo de 0,95
PVP_ABAIXO_DA_MEDIA = 0.10       # caminho 4b: P/VP 10% abaixo da própria média


def _sem_fuso(serie):
    if serie is None:
        return pd.Series(dtype=float)
    if getattr(serie.index, "tz", None) is not None:
        serie = serie.copy()
        serie.index = serie.index.tz_localize(None)
    return serie.sort_index()


def _vp_com_defasagem(valor_patrimonial: pd.Series | None, datas) -> pd.Series:
    """Valor patrimonial por cota conhecido em cada data: o do informe de
    referência do mês ANTERIOR (o informe sai cerca de um mês depois),
    para não usar no passado um dado que ainda não tinha sido publicado."""
    if valor_patrimonial is None or len(valor_patrimonial) == 0:
        return pd.Series(np.nan, index=datas)
    vp = _sem_fuso(valor_patrimonial).dropna()
    vp = vp[vp > 0]
    vp = vp.groupby(vp.index.to_period("M")).last()
    saida = []
    for t in datas:
        ref = pd.Timestamp(t).to_period("M") - 1
        disponiveis = vp[vp.index <= ref]
        saida.append(float(disponiveis.iloc[-1]) if len(disponiveis) else np.nan)
    return pd.Series(saida, index=datas)


def _ipca_por_mes(ipca_mensal: pd.Series | None) -> pd.Series:
    """IPCA do mês (fração, ex.: 0,0045) indexado por período mensal."""
    if ipca_mensal is None or len(ipca_mensal) == 0:
        return pd.Series(dtype=float)
    s = _sem_fuso(ipca_mensal).dropna()
    return s.groupby(s.index.to_period("M")).last()


def ipca_acumulado(ipca: pd.Series, fim: pd.Period, meses: int) -> float:
    """IPCA acumulado nos ``meses`` meses que terminam em ``fim`` (inclusive);
    NaN se faltar algum mês."""
    janela = [fim - k for k in range(meses)]
    if not all(p in ipca.index for p in janela):
        return np.nan
    return float(np.prod([1 + ipca[p] for p in janela]) - 1)


def renda_acima_da_inflacao(df: pd.DataFrame, vp: pd.Series, ipca_mensal: pd.Series | None,
                            queda_maxima: float = QUEDA_MAXIMA_RENDIMENTOS):
    """Etapa C (04/11/2026, regra fixada antes de rodar). Num FII de papel
    atrelado ao IPCA, o rendimento nominal = juro real dos CRIs + correção
    pela inflação: quando o IPCA desacelera, o rendimento cai sem que o
    fundo piore. Por isso a estabilidade é medida também na RENDA ACIMA DA
    INFLAÇÃO: rendimentos de 12 meses ÷ valor patrimonial (CVM) − IPCA de 12
    meses, com 2 meses de defasagem (a correção dos CRIs usa o IPCA de
    ~2 meses antes). Estável em termos reais: essa renda não caiu mais de 10%
    sobre os 12 meses anteriores NEM nos últimos 3 meses (anualizados) —
    a mesma régua da estabilidade nominal. Devolve (renda_real_12m, estavel_real)
    como arrays (NaN quando falta IPCA, valor patrimonial ou histórico)."""
    n = len(df)
    ipca = _ipca_por_mes(ipca_mensal)
    if ipca.empty or vp is None:
        return np.full(n, np.nan), np.full(n, np.nan)
    per = pd.to_datetime(df["data"]).dt.to_period("M").tolist()
    vp_por_mes = dict(zip(per, np.asarray(vp, dtype=float)))

    def vp_em(p):
        v = vp_por_mes.get(p, np.nan)
        return v if v and v > 0 else np.nan

    renda12, estavel = [], []
    for i, p in enumerate(per):
        linha = df.iloc[i]
        r12 = linha["div12"] / vp_em(p) - ipca_acumulado(ipca, p - 2, 12)
        r12_ant = linha["div12_anterior"] / vp_em(p - 12) - ipca_acumulado(ipca, p - 14, 12)
        i3 = ipca_acumulado(ipca, p - 2, 3)
        r3 = linha["div3m_anualizado"] / vp_em(p) - ((1 + i3) ** 4 - 1)
        r_ref = linha["div12_ate_3m_atras"] / vp_em(p - 3) - ipca_acumulado(ipca, p - 5, 12)
        renda12.append(r12)
        valores = (r12, r12_ant, r3, r_ref)
        if any(pd.isna(v) for v in valores):
            estavel.append(np.nan)
            continue
        ok = (r12_ant > 0 and r12 >= (1 - queda_maxima) * r12_ant
              and r_ref > 0 and r3 >= (1 - queda_maxima) * r_ref)
        estavel.append(1.0 if ok else 0.0)
    return np.array(renda12, dtype=float), np.array(estavel, dtype=float)


def avaliar_fii_mes_a_mes(precos: pd.Series, dividendos: pd.Series, indice_cdi: pd.Series,
                          juro_real: pd.Series | None = None,
                          valor_patrimonial: pd.Series | None = None,
                          ipca_mensal: pd.Series | None = None, papel: bool = False,
                          queda_maxima: float = QUEDA_MAXIMA_RENDIMENTOS) -> pd.DataFrame:
    """Uma linha por fim de mês avaliável de um FII: yield, média do próprio
    yield, prêmio sobre o Tesouro IPCA+, P/VP (se ``valor_patrimonial`` vier,
    da CVM, na mesma escala dos preços), critérios (caminhos 2, 3 e 4),
    retorno dos 12 meses seguintes e CDI líquido do mesmo período.
    Com ``papel=True`` e ``ipca_mensal`` (IPCA do mês, em fração), calcula
    também a estabilidade da renda descontada a inflação e as variantes
    "_real" dos caminhos 2, 4a e 4c (ver ``renda_acima_da_inflacao``).
    ``queda_maxima``: queda tolerada para os rendimentos serem "estáveis"
    (10% na regra fixada; 5% e 15% só na checagem de robustez)."""
    p = _sem_fuso(precos).dropna()
    d = _sem_fuso(dividendos)
    if len(p) < 60 or len(d) == 0:
        return pd.DataFrame()
    cdi = indice_cdi.reindex(p.index).ffill() if indice_cdi is not None else None
    juro = None
    if juro_real is not None and len(juro_real):
        juro = _sem_fuso(juro_real).reindex(_sem_fuso(juro_real).index.union(p.index)).ffill().reindex(p.index)

    def soma_div(ini, fim):
        return float(d[(d.index > ini) & (d.index <= fim)].sum()) if len(d) else 0.0

    def n_pag(ini, fim):
        return int(((d.index > ini) & (d.index <= fim)).sum()) if len(d) else 0

    def completa(ini, fim):
        # O histórico de rendimentos de FIIs no Yahoo tem lacunas (meses sem
        # registro). Uma janela com lacuna subestima o yield e contaminaria a média
        # e o resultado; por isso só janelas com ao menos 10 pagamentos contam.
        return n_pag(ini, fim) >= MIN_PAGAMENTOS_12M

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
        if preco <= 0:
            continue
        # uma linha por MÊS do calendário, mesmo com lacuna (yield vazio): assim a
        # média de 24 meses exige 24 meses seguidos válidos, sem "pular" lacunas
        janela_ok = completa(t - m(months=12), t)
        div12 = soma_div(t - m(months=12), t)
        anterior_ok = t >= inicio_dados + m(months=24) and completa(t - m(months=24), t - m(months=12))
        referencia_ok = t >= inicio_dados + m(months=15) and completa(t - m(months=15), t - m(months=3))
        linhas.append({
            "data": t, "preco": preco, "div12": div12 if janela_ok else np.nan,
            "yield": div12 / preco if janela_ok and div12 > 0 else np.nan,
            "div12_anterior": soma_div(t - m(months=24), t - m(months=12)) if anterior_ok else np.nan,
            # últimos 3 meses (anualizados) x os 12 meses que terminam 3 meses antes:
            # pega cortes recentes que a soma de 12 meses ainda esconde (yield trap)
            "div3m_anualizado": soma_div(t - m(months=3), t) * 4,
            "div12_ate_3m_atras": soma_div(t - m(months=15), t - m(months=3)) if referencia_ok else np.nan,
            "juro_real": float(juro.loc[t]) if juro is not None and pd.notna(juro.loc[t]) else None})
    df = pd.DataFrame(linhas)
    if df.empty:
        return df
    df["premio"] = df["yield"] - df["juro_real"].astype(float)
    df["yield_medio_24m"] = df["yield"].shift(1).rolling(MESES_MEDIA, min_periods=MESES_MEDIA).mean()
    df["premio_medio_24m"] = df["premio"].shift(1).rolling(MESES_MEDIA, min_periods=MESES_MEDIA).mean()
    df["estavel"] = ((df["div12_anterior"] > 0)
                     & (df["div12"] >= (1 - queda_maxima) * df["div12_anterior"])
                     & (df["div3m_anualizado"] >= (1 - queda_maxima) * df["div12_ate_3m_atras"]))
    sem_comparacao = df["div12_anterior"].isna() | df["yield"].isna() | df["div12_ate_3m_atras"].isna()
    # critérios como 1/0, e vazio (NaN) enquanto não há 24 meses de histórico para a média
    crit_y = (df["yield"] >= (1 + DESCONTO_MINIMO_YIELD) * df["yield_medio_24m"]) & df["estavel"]
    crit_p = (df["premio"] >= df["premio_medio_24m"] + PREMIO_EXTRA_MINIMO) & df["estavel"]
    df["criterio_yield"] = np.where(df["yield_medio_24m"].isna() | sem_comparacao, np.nan, crit_y.astype(float))
    df["criterio_premio"] = np.where(df["premio_medio_24m"].isna() | sem_comparacao, np.nan, crit_p.astype(float))

    # caminho 4: P/VP (valor patrimonial da CVM, com 1 mês de defasagem)
    vp = _vp_com_defasagem(valor_patrimonial, df["data"])
    df["pvp"] = (df["preco"] / vp.values).where(vp.values > 0)
    df["pvp_medio_24m"] = df["pvp"].shift(1).rolling(MESES_MEDIA, min_periods=MESES_MEDIA).mean()
    sem_pvp = df["pvp"].isna() | sem_comparacao
    crit_4a = (df["pvp"] < PVP_DESCONTO) & df["estavel"]
    crit_4b = (df["pvp"] <= (1 - PVP_ABAIXO_DA_MEDIA) * df["pvp_medio_24m"]) & df["estavel"]
    crit_4c = crit_y & (df["pvp"] < PVP_DESCONTO)
    df["criterio_pvp"] = np.where(sem_pvp, np.nan, crit_4a.astype(float))
    df["criterio_pvp_historico"] = np.where(sem_pvp | df["pvp_medio_24m"].isna(), np.nan, crit_4b.astype(float))
    df["criterio_yield_pvp"] = np.where(sem_pvp | df["yield_medio_24m"].isna(), np.nan, crit_4c.astype(float))

    # Etapa C: estabilidade descontada a inflação (só FIIs de papel atrelados a índice)
    df["renda_real_12m"], df["estavel_real"] = renda_acima_da_inflacao(df, vp, ipca_mensal, queda_maxima) if papel else (np.nan, np.nan)
    ampliado = df["estavel"] | (df["estavel_real"] == 1.0)
    crit_y_r = (df["yield"] >= (1 + DESCONTO_MINIMO_YIELD) * df["yield_medio_24m"]) & ampliado
    crit_4a_r = (df["pvp"] < PVP_DESCONTO) & ampliado
    crit_4c_r = crit_y_r & (df["pvp"] < PVP_DESCONTO)
    df["estavel_ampliado"] = ampliado
    df["criterio_yield_real"] = np.where(df["yield_medio_24m"].isna() | sem_comparacao, np.nan, crit_y_r.astype(float))
    df["criterio_pvp_real"] = np.where(sem_pvp, np.nan, crit_4a_r.astype(float))
    df["criterio_yield_pvp_real"] = np.where(sem_pvp | df["yield_medio_24m"].isna(), np.nan, crit_4c_r.astype(float))

    # resultado: retorno total em 12 meses x CDI líquido do mesmo período
    ultimo = p.index[-1]
    ret, cdi_liq, lacuna_futuro = [], [], []
    for t, preco in zip(df["data"], df["preco"]):
        fim = t + pd.DateOffset(months=12)
        # sem os 12 meses seguintes, ou com lacuna de rendimentos neles (não dá para
        # distinguir lacuna da fonte de corte real), o caso não é avaliado
        lacuna_futuro.append(fim <= ultimo and not completa(t, fim))
        if fim > ultimo or lacuna_futuro[-1]:
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
    df["lacuna_futuro"] = lacuna_futuro
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
        # Totais por FII (casos e acertos, no geral e com o critério): somar os
        # totais dos FIIs sorteados dá exatamente as mesmas taxas que juntar as
        # linhas, sem copiar a tabela 2.000 vezes (de minutos para menos de 1 s).
        sup = base["superou_cdi"].astype(float)
        por_fii = pd.DataFrame({"n": 1.0, "s": sup, "nc": base[coluna].astype(float),
                                "sc": sup * base[coluna].astype(float), "ticker": base["ticker"]}
                               ).groupby("ticker", sort=True).sum()
        n, s, nc, sc = (por_fii[c].to_numpy() for c in ("n", "s", "nc", "sc"))
        k = len(por_fii)
        difs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(k, size=k, replace=True)
            n_com = nc[idx].sum()
            if n_com:
                difs.append(sc[idx].sum() / n_com - s[idx].sum() / n[idx].sum())
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
