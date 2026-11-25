"""Visão geral da carteira: a carteira mês a mês.

Reconstrói, para cada ativo, a posição no fim de cada mês a partir das
operações da planilha (as mesmas do extrato, com eventos aplicados) e soma:
patrimônio, valor investido (custo das posições abertas), aportes, vendas e
proventos recebidos. Com isso calcula a rentabilidade pelo método das cotas
(Dietz modificado, mês a mês, encadeado), comparável ao CDI acumulado.

Posições encerradas entram nos meses em que estavam abertas; a venda sai da
carteira como resgate (o painel não sabe para onde foi o dinheiro).

Escala de preços e proventos: o Yahoo ajusta preços e proventos por
desdobramento/grupamento (e às vezes bonificação). A quantidade da planilha é
a da época; para multiplicar as duas, o valor ajustado é convertido para a
escala da época pelo produto dos fatores dos eventos POSTERIORES à data
(``fator_escala``). Preço da base B3 (COTAHIST) já é da época: fator 1.
Este módulo não faz I/O: recebe as séries prontas.
"""
from __future__ import annotations

import pandas as pd

COLUNAS = ["quantidade", "patrimonio", "custo", "aportes", "vendas", "proventos"]


def _sem_fuso(indice) -> pd.DatetimeIndex:
    indice = pd.DatetimeIndex(indice)
    return indice.tz_localize(None) if indice.tz is not None else indice


def fator_escala(eventos: pd.Series | None, data) -> float:
    """Produto dos fatores dos eventos depois de ``data``: multiplica um valor
    por cota ajustado (escala de hoje) para chegar à escala da época."""
    if eventos is None or len(eventos) == 0:
        return 1.0
    ev = pd.Series(eventos, dtype=float)
    ev.index = _sem_fuso(ev.index)
    fator = 1.0
    for d, f in ev.items():
        if d > pd.Timestamp(data) and f and f > 0:
            fator *= float(f)
    return fator


def meses_entre(inicio, fim) -> pd.PeriodIndex:
    return pd.period_range(pd.Period(inicio, "M"), pd.Period(fim, "M"), freq="M")


def preco_no_mes(precos: pd.Series | None, mes: pd.Period) -> float | None:
    """Último preço até o fim do mês (None se não houver nenhum até ali)."""
    if precos is None or len(precos) == 0:
        return None
    s = pd.Series(precos, dtype=float).dropna()
    s.index = _sem_fuso(s.index)
    s = s[s.index <= mes.end_time]
    return float(s.iloc[-1]) if len(s) else None


def linha_do_tempo_ativo(ops: list[dict], meses: pd.PeriodIndex, precos: pd.Series | None,
                         proventos: pd.Series | None = None, eventos: pd.Series | None = None,
                         precos_ajustados: bool = True) -> pd.DataFrame:
    """Um ativo mês a mês (``COLUNAS``), na moeda do ativo.

    ``ops``: operações de ``operacoes_do_ticker`` (data, tipo, quantidade,
    preco). ``precos``: fechamentos (diários ou mensais); ``precos_ajustados``
    False quando vêm da base B3, já na escala da época. ``proventos``: valor
    por cota na data-ex, na escala do Yahoo (ajustada)."""
    ops = sorted(ops, key=lambda o: pd.Timestamp(o["data"]))
    divs = pd.Series(dtype=float) if proventos is None else pd.Series(proventos, dtype=float).dropna()
    divs.index = _sem_fuso(divs.index)
    linhas = []
    q = custo = 0.0
    i = 0
    for mes in meses:
        fim = mes.end_time
        aportes = vendas = recebido = 0.0
        # proventos do mês: quantidade mantida ANTES da data-ex (operações do dia não contam)
        for data, valor in divs[(divs.index >= mes.start_time) & (divs.index <= fim)].items():
            q_antes = _quantidade_antes(ops, data)
            if q_antes > 1e-9:
                recebido += float(valor) * fator_escala(eventos, data) * q_antes
        while i < len(ops) and pd.Timestamp(ops[i]["data"]) <= fim:
            op = ops[i]
            i += 1
            qtd, preco = float(op["quantidade"]), float(op.get("preco") or 0.0)
            tipo = op["tipo"]
            if tipo in ("compra", "bonificacao"):
                q += qtd
                custo += qtd * preco
                if tipo == "compra":  # bonificação: custo atribuído (IR), sem saída de caixa
                    aportes += qtd * preco
            elif tipo == "venda":
                pm = custo / q if q > 1e-12 else 0.0
                q -= qtd
                custo -= qtd * pm
                vendas += qtd * preco
            elif tipo == "desdobramento":
                q += qtd
            elif tipo == "grupamento":
                q -= qtd
            if q < 1e-9:
                q = custo = 0.0
        patrimonio = 0.0
        if q > 1e-9:
            p = preco_no_mes(precos, mes)
            if p is not None:
                patrimonio = q * p * (fator_escala(eventos, fim) if precos_ajustados else 1.0)
        linhas.append({"quantidade": q, "patrimonio": patrimonio, "custo": custo,
                       "aportes": aportes, "vendas": vendas, "proventos": recebido})
    return pd.DataFrame(linhas, index=meses, columns=COLUNAS)


def _quantidade_antes(ops: list[dict], data) -> float:
    q = 0.0
    for op in ops:
        if pd.Timestamp(op["data"]) >= pd.Timestamp(data):
            break
        if op["tipo"] in ("compra", "bonificacao", "desdobramento"):
            q += float(op["quantidade"])
        elif op["tipo"] in ("venda", "grupamento"):
            q -= float(op["quantidade"])
    return max(q, 0.0)


def somar(linhas: dict[str, pd.DataFrame], tickers=None, cambio: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Soma os ativos escolhidos mês a mês. ``cambio``: ticker -> série
    mensal (PeriodIndex) para converter para reais (ativos dos EUA)."""
    escolhidos = [t for t in linhas if tickers is None or t in tickers]
    if not escolhidos:
        return pd.DataFrame(columns=COLUNAS[1:])
    total = None
    for t in escolhidos:
        df = linhas[t][COLUNAS[1:]].copy()
        if cambio and t in cambio and cambio[t] is not None:
            df = df.mul(cambio[t].reindex(df.index).ffill().bfill(), axis=0)
        total = df if total is None else total.add(df, fill_value=0.0)
    return total


def rentabilidade_cotas(df: pd.DataFrame, patrimonio_inicial: float = 0.0) -> pd.Series:
    """Rentabilidade acumulada (0,10 = +10%) mês a mês, pelo método das cotas.

    Cada mês: r = (V1 + vendas + P − V0 − aportes) / (V0 + aportes): o
    ganho do mês sobre o capital empregado nele. O que foi comprado rende a
    partir do preço de compra; o que foi vendido rende até o preço de venda
    (estava em V0). P = proventos recebidos. Os meses são encadeados, então
    aportes e vendas não contam como rentabilidade. (Com a convenção de
    "meio do mês", uma compra no fim do mês dobrava o ganho aparente: o
    painel mostrava +25% em jul/2019.)"""
    v0 = float(patrimonio_inicial)
    acumulado, cota = [], 1.0
    for _, linha in df.iterrows():
        aportes, vendas = float(linha["aportes"]), float(linha["vendas"])
        base = v0 + aportes
        ganho = float(linha["patrimonio"]) + vendas + float(linha["proventos"]) - v0 - aportes
        r = ganho / base if base > 1e-6 else 0.0
        cota *= 1 + r
        acumulado.append(cota - 1)
        v0 = float(linha["patrimonio"])
    return pd.Series(acumulado, index=df.index)


def cdi_acumulado(cdi_diario_pct: pd.Series | None, meses: pd.PeriodIndex) -> pd.Series | None:
    """CDI acumulado (0,10 = +10%) no fim de cada mês, a partir do início do
    primeiro mês. ``cdi_diario_pct``: série do Banco Central (% ao dia útil)."""
    if cdi_diario_pct is None or len(cdi_diario_pct) == 0 or len(meses) == 0:
        return None
    s = pd.Series(cdi_diario_pct, dtype=float).dropna().sort_index()
    s.index = _sem_fuso(s.index)
    s = s[s.index >= meses[0].start_time]
    fator = (1 + s / 100).cumprod()
    saida = []
    for mes in meses:
        parte = fator[fator.index <= mes.end_time]
        saida.append(float(parte.iloc[-1]) - 1 if len(parte) else 0.0)
    return pd.Series(saida, index=meses)


def resumo_periodo(df: pd.DataFrame, patrimonio_inicial: float = 0.0) -> dict:
    """Números do topo da aba para o período de ``df`` (já filtrado)."""
    if df is None or df.empty:
        return {}
    rent = rentabilidade_cotas(df, patrimonio_inicial)
    return {"patrimonio_final": float(df["patrimonio"].iloc[-1]),
            "patrimonio_inicial": float(patrimonio_inicial),
            "aportes": float(df["aportes"].sum()), "vendas": float(df["vendas"].sum()),
            "proventos": float(df["proventos"].sum()),
            "proventos_mes": float(df["proventos"].sum()) / len(df),
            "rentabilidade": float(rent.iloc[-1])}


def tir_anual(df: pd.DataFrame, patrimonio_inicial: float = 0.0) -> float | None:
    """Rentabilidade ponderada pelo dinheiro (TIR) ao ano. Fluxos mensais do
    ponto de vista do investidor: sai o patrimônio inicial e os aportes,
    entram as vendas e os proventos, e no fim o patrimônio final. None se não
    houver solução (sem dinheiro investido, por exemplo)."""
    if df is None or df.empty:
        return None
    fluxos = [-float(patrimonio_inicial)]
    for _, linha in df.iterrows():
        fluxos.append(float(linha["vendas"]) + float(linha["proventos"]) - float(linha["aportes"]))
    fluxos[-1] += float(df["patrimonio"].iloc[-1])

    def vpl(taxa):
        return sum(f / (1 + taxa) ** i for i, f in enumerate(fluxos))
    if not any(f < 0 for f in fluxos) or not any(f > 0 for f in fluxos):
        return None
    lo, hi = -0.99, 1.0
    if vpl(lo) * vpl(hi) > 0:
        return None
    for _ in range(200):  # bisseção na taxa mensal
        meio = (lo + hi) / 2
        if vpl(lo) * vpl(meio) <= 0:
            hi = meio
        else:
            lo = meio
    return (1 + (lo + hi) / 2) ** 12 - 1


def resultado_em_reais(df: pd.DataFrame, patrimonio_inicial: float = 0.0) -> float:
    """Ganho do período em dinheiro: patrimônio final − inicial − aportes
    líquidos (compras − vendas) + proventos."""
    if df is None or df.empty:
        return 0.0
    return (float(df["patrimonio"].iloc[-1]) - float(patrimonio_inicial)
            - float(df["aportes"].sum()) + float(df["vendas"].sum()) + float(df["proventos"].sum()))


def por_ano(df: pd.DataFrame, patrimonio_inicial: float = 0.0,
            cdi_diario_pct: pd.Series | None = None) -> pd.DataFrame:
    """Uma linha por ano do período: rentabilidade (cotas), CDI do ano,
    resultado em dinheiro, proventos (total e média por mês)."""
    linhas = []
    v0 = float(patrimonio_inicial)
    for ano, parte in df.groupby(df.index.year):
        rent = float(rentabilidade_cotas(parte, v0).iloc[-1])
        cdi = cdi_acumulado(cdi_diario_pct, parte.index)
        linhas.append({"ano": int(ano), "meses": len(parte), "rentabilidade": rent,
                       "cdi": float(cdi.iloc[-1]) if cdi is not None else None,
                       "resultado": resultado_em_reais(parte, v0),
                       "proventos": float(parte["proventos"].sum()),
                       "proventos_mes": float(parte["proventos"].sum()) / len(parte)})
        v0 = float(parte["patrimonio"].iloc[-1])
    return pd.DataFrame(linhas)


def indice_acumulado(serie: pd.Series | None, meses: pd.PeriodIndex) -> pd.Series | None:
    """Variação acumulada de um índice (0,10 = +10%) no fim de cada mês,
    a partir do fechamento do mês ANTERIOR ao primeiro (mesma base da
    rentabilidade da carteira). None se faltar a base."""
    if serie is None or len(serie) == 0 or len(meses) == 0:
        return None
    base = preco_no_mes(serie, meses[0] - 1)
    if base is None or base <= 0:
        return None
    valores = [preco_no_mes(serie, m) for m in meses]
    return pd.Series([v / base - 1 if v is not None else float("nan") for v in valores], index=meses)


def indice_por_ano(serie: pd.Series | None, anos: list[int], ultimo_mes: pd.Period | None = None) -> dict[int, float]:
    """Variação do índice em cada ano (do fim do ano anterior ao fim do ano,
    ou até ``ultimo_mes`` no ano corrente)."""
    saida = {}
    if serie is None or len(serie) == 0:
        return saida
    for a in anos:
        fim = pd.Period(f"{a}-12", "M")
        if ultimo_mes is not None and fim > ultimo_mes:
            fim = ultimo_mes
        base, valor = preco_no_mes(serie, pd.Period(f"{a - 1}-12", "M")), preco_no_mes(serie, fim)
        if base and valor:
            saida[a] = valor / base - 1
    return saida
