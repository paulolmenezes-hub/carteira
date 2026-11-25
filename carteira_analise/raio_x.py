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
    """Maior queda (do topo ao fundo, com as datas) e volatilidade anualizada
    nos últimos 12 meses, a partir dos fechamentos diários."""
    if precos is None or len(precos) < 30:
        return None
    s = _sem_fuso(precos)
    s = s[s.index >= s.index[-1] - pd.DateOffset(months=12)]
    if len(s) < 30:
        return None
    recuo = s / s.cummax() - 1
    queda = float(recuo.min())
    fundo = recuo.idxmin()
    topo = s[s.index <= fundo].idxmax()
    vol = float(s.pct_change().dropna().std() * math.sqrt(252))
    return {"maior_queda": queda, "volatilidade": vol,
            "data_topo": topo if queda < 0 else None, "data_fundo": fundo if queda < 0 else None}


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


# Sinal de leitura dos indicadores no cabeçalho do Raio-X: "bom" (azul),
# "ruim" (vermelho) ou None (preto: neutro ou sem referência). Os limiares são
# os do motor de pontuação (pontuacao.py) e do Guia de Indicadores; não é
# recomendação, só a leitura de cada número isolado.
LIMIARES_ACAO = {
    # nome: (sentido, limiar bom, limiar ruim)
    "pvp": ("menor", 1.5, 4.0),
    "roe": ("maior", 0.12, 0.0),
    "roic": ("maior", 0.15, 0.0),
    "margem_liquida": ("maior", 0.10, 0.0),
    "divida_bruta_patrimonio": ("menor", 0.5, 1.5),
    "liquidez_corrente": ("maior", 1.5, 1.0),
    "crescimento_receita_5a": ("maior", 0.0, 0.0),
}
LIMIARES_FII = {
    "pvp": ("menor", 1.0, 1.2),
    "vacancia_media": ("menor", 0.05, 0.10),
}
_ZERO_E_AUSENTE = {"divida_bruta_patrimonio", "liquidez_corrente"}  # bancos: o Fundamentus mostra "-"


def sinal(nome: str, valor, limiares: dict = LIMIARES_ACAO) -> str | None:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)) or nome not in limiares:
        return None
    if nome in _ZERO_E_AUSENTE and valor == 0:
        return None
    sentido, bom, ruim = limiares[nome]
    if sentido == "menor":
        return "bom" if valor < bom else "ruim" if valor > ruim else None
    return "bom" if valor > bom else "ruim" if valor < ruim else None


def sinal_pl(pl, leitura: str | None) -> str | None:
    """P/L: prejuízo é ruim; com lucro, vale a comparação com a referência
    (segmento, setor ou bolsa): abaixo = bom, acima = ruim."""
    if pl is not None and not pd.isna(pl) and pl <= 0:
        return "ruim"
    txt = (leitura or "").lower()
    return "bom" if txt.startswith("abaixo") else "ruim" if txt.startswith("acima") else None


# Aviso de tendência de médio prazo rompida: informação para reavaliar a
# posição, não sinal de venda.
PREGOES_ABAIXO_MM200 = 20
JANELA_CRUZAMENTO = 60  # pregões: cruzamento da MM50 abaixo da MM200 conta como recente


def tendencia_rompida(precos: pd.Series | None) -> dict | None:
    """Devolve o aviso se, no fim da série, (a) o preço fecha abaixo da média
    de 200 dias há ``PREGOES_ABAIXO_MM200`` pregões seguidos ou mais, e/ou (b)
    a média de 50 dias cruzou para baixo da de 200 nos últimos
    ``JANELA_CRUZAMENTO`` pregões e continua abaixo. None se nada disso.

    Chaves: ``abaixo_desde`` (1º pregão da sequência abaixo da MM200),
    ``pregoes_abaixo``, ``cruzamento`` (data do cruzamento, ou None),
    ``preco``, ``mm200`` e ``distancia`` (preço / MM200 − 1)."""
    if precos is None or len(precos) < 220:
        return None
    s = _sem_fuso(precos).dropna()
    mm200, mm50 = s.rolling(200).mean(), s.rolling(50).mean()
    validos = mm200.notna()
    s, mm200, mm50 = s[validos], mm200[validos], mm50[validos]
    if s.empty:
        return None
    abaixo = (s < mm200).to_numpy()
    n = 0
    for v in abaixo[::-1]:
        if not v:
            break
        n += 1
    cruzamento = None
    acima_antes = (mm50 >= mm200).to_numpy()
    if not acima_antes[-1]:
        for i in range(len(acima_antes) - 1, max(len(acima_antes) - JANELA_CRUZAMENTO, 0) - 1, -1):
            if i > 0 and acima_antes[i - 1] and not acima_antes[i]:
                cruzamento = s.index[i]
                break
    if n < PREGOES_ABAIXO_MM200 and cruzamento is None:
        return None
    return {"abaixo_desde": s.index[-n] if n else None, "pregoes_abaixo": n, "cruzamento": cruzamento,
            "preco": float(s.iloc[-1]), "mm200": float(mm200.iloc[-1]),
            "distancia": float(s.iloc[-1] / mm200.iloc[-1] - 1)}


JANELA_RETOMADA = 63  # pregões (~3 meses): depois disso a alta já é a tendência normal, sem aviso


def _sequencia_final(flags) -> int:
    n = 0
    for v in flags[::-1]:
        if not v:
            break
        n += 1
    return n


def tendencia_retomada(precos: pd.Series | None) -> dict | None:
    """O espelho de ``tendencia_rompida``: avisa a VIRADA para cima, não o
    estado de alta. (a) O preço fecha acima da média de 200 dias há 20 a 63
    pregões seguidos, depois de pelo menos 20 abaixo; e/ou (b) a média de 50
    dias cruzou para cima da de 200 nos últimos ``JANELA_CRUZAMENTO`` pregões
    e continua acima. Mesmas chaves, com ``acima_desde`` e ``pregoes_acima``."""
    if precos is None or len(precos) < 220:
        return None
    s = _sem_fuso(precos).dropna()
    mm200, mm50 = s.rolling(200).mean(), s.rolling(50).mean()
    validos = mm200.notna()
    s, mm200, mm50 = s[validos], mm200[validos], mm50[validos]
    if s.empty:
        return None
    acima = (s > mm200).to_numpy()
    n = _sequencia_final(acima)
    antes = _sequencia_final(~acima[:len(acima) - n]) if n < len(acima) else 0
    virou = PREGOES_ABAIXO_MM200 <= n <= JANELA_RETOMADA and antes >= PREGOES_ABAIXO_MM200
    cruzamento = None
    abaixo_antes = (mm50 <= mm200).to_numpy()
    if not abaixo_antes[-1]:
        for i in range(len(abaixo_antes) - 1, max(len(abaixo_antes) - JANELA_CRUZAMENTO, 0) - 1, -1):
            if i > 0 and abaixo_antes[i - 1] and not abaixo_antes[i]:
                cruzamento = s.index[i]
                break
    if not virou and cruzamento is None:
        return None
    return {"acima_desde": s.index[-n] if virou else None, "pregoes_acima": n if virou else 0,
            "cruzamento": cruzamento, "preco": float(s.iloc[-1]), "mm200": float(mm200.iloc[-1]),
            "distancia": float(s.iloc[-1] / mm200.iloc[-1] - 1)}


def tendencia_do_ativo(precos: pd.Series | None) -> dict | None:
    """Rompimento (``sentido`` "baixa") ou retomada ("alta") da tendência de
    médio prazo; None se nenhum dos dois."""
    t = tendencia_rompida(precos)
    if t:
        return {**t, "sentido": "baixa"}
    t = tendencia_retomada(precos)
    return {**t, "sentido": "alta"} if t else None
