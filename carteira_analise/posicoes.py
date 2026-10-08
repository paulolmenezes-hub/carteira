"""Posição sugerida por ativo no painel — liga a regra de faixas de preço
(``regra_posicao``) à carteira importada da planilha (abas Resumo +
Operações), usando a mesma fonte de dados do resto do painel.

``fonte`` segue o mesmo protocolo de ``processar_carteira_combinada``:
precisa expor ``baixar_precos(ticker, periodo)`` e
``baixar_dividendos(ticker)``. Sem I/O de rede direto aqui, o que mantém
o módulo testável com fonte falsa.
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from .planilha import (
    _TIPO_PARA_GRUPO,
    _construir_operacao_da_posicao_inicial,
    interpretar_quantidade_operacao,
    normalizar_ticker,
)
from .regra_posicao import (
    decidir_posicao,
    estado_a_partir_de_operacoes,
    filtro_fundamentos_compra,
    montar_card_posicao,
    quantidade_para_acao,
)


def _sem_fuso(serie: pd.Series) -> pd.Series:
    if serie is None:
        return pd.Series(dtype=float)
    if getattr(serie.index, "tz", None) is not None:
        serie = serie.copy()
        serie.index = serie.index.tz_localize(None)
    return serie


def operacoes_por_ticker(df_resumo: pd.DataFrame | None,
                         df_operacoes: pd.DataFrame | None) -> dict[str, list[dict]]:
    """Mesma combinação Resumo + Operações de ``processar_carteira_combinada``
    (posição inicial primeiro, operações em cima), no formato de dict usado
    pela regra. Linhas inválidas são ignoradas aqui — o painel já avisa
    sobre elas na leitura da planilha e no dashboard."""
    ops: dict[str, list[dict]] = {}
    if df_resumo is not None:
        for _, row in df_resumo.iterrows():
            ticker, _ = normalizar_ticker(row["ticker"])
            try:
                op = _construir_operacao_da_posicao_inicial(row)
            except (ValueError, TypeError):
                continue
            ops.setdefault(ticker, []).append({
                "data": pd.Timestamp(op.data), "tipo": op.tipo,
                "quantidade": float(op.quantidade), "preco": float(op.preco)})
    if df_operacoes is not None and not df_operacoes.empty:
        for _, row in df_operacoes.iterrows():
            ticker, _ = normalizar_ticker(row["ticker"])
            tipo_bruto = str(row["tipo"]).strip().lower()
            if tipo_bruto in ("compra", "buy", "c"):
                tipo = "compra"
            elif tipo_bruto in ("venda", "sell", "v"):
                tipo = "venda"
            else:
                continue
            quantidade, _ = interpretar_quantidade_operacao(tipo, row["quantidade"])
            if quantidade is None:
                continue
            try:
                preco = float(row["preco"])
                if not preco > 0:
                    continue
                ops.setdefault(ticker, []).append({
                    "data": pd.Timestamp(row["data"]), "tipo": tipo, "quantidade": quantidade, "preco": preco})
            except (ValueError, TypeError):
                continue
    return ops


def renda_12m_por_cota(dividendos: pd.Series | None, data_ref) -> float | None:
    """Soma dos proventos por cota pagos nos 12 meses até ``data_ref``."""
    d = _sem_fuso(dividendos)
    if d is None or d.empty:
        return None
    data_ref = pd.Timestamp(data_ref)
    soma = float(d[(d.index > data_ref - pd.Timedelta(days=365)) & (d.index <= data_ref)].sum())
    return soma if soma > 0 else None


def fundamentos_de_resultado(resultado) -> dict:
    """Extrai o dicionário de fundamentos do resultado de ``analisar_ativo``
    (atributo ``fund``, ou ``fundamentos``), aceitando dict, dataclass ou
    objeto simples. Sem fundamentos, devolve {} — dado ausente não bloqueia
    a compra (mesmo critério do motor de pontuação)."""
    if resultado is None:
        return {}
    for nome in ("fund", "fundamentos"):
        valor = getattr(resultado, nome, None)
        if valor is None:
            continue
        if isinstance(valor, dict):
            return dict(valor)
        if dataclasses.is_dataclass(valor):
            return dataclasses.asdict(valor)
        if hasattr(valor, "__dict__"):
            return dict(vars(valor))
    return {}


_ORDEM_MERCADO = {"B3": 0, "EUA": 1}
_ORDEM_GRUPO = {"Ações": 0, "FIIs": 1, "ETF": 2, "Outros": 3}


def chave_ordem_card(card: dict) -> tuple:
    """Ordena os cards por mercado (B3, depois EUA), tipo de ativo (Ações,
    FIIs, ETF — mesmos grupos do extrato do Dashboard) e ticker."""
    return (_ORDEM_MERCADO.get(card.get("mercado"), 9), _ORDEM_GRUPO.get(card.get("grupo"), 9),
            card.get("ticker", ""))


def gerar_posicoes(df_resumo, df_operacoes, fonte, tipos: dict[str, str] | None = None,
                   fundamentos: dict[str, dict] | None = None,
                   periodo: str = "5y") -> tuple[list[dict], list[str]]:
    """Aplica a regra de faixas a cada posição ABERTA da carteira. Devolve
    (cards, avisos): um card por ativo (ver ``montar_card_posicao``), com as
    chaves extras ``ticker``, ``quantidade``, ``mercado`` e ``grupo``,
    ordenados por mercado, tipo de ativo e ticker (``chave_ordem_card``)."""
    tipos = tipos or {}
    fundamentos = fundamentos or {}
    cards: list[dict] = []
    avisos: list[str] = []

    for ticker, ops in sorted(operacoes_por_ticker(df_resumo, df_operacoes).items()):
        try:
            precos = _sem_fuso(fonte.baixar_precos(ticker, periodo)).dropna()
        except Exception:
            precos = pd.Series(dtype=float)
        if precos.empty:
            avisos.append(f"{ticker}: sem cotação disponível — posição não avaliada.")
            continue
        preco = float(precos.iloc[-1])
        data_ref = precos.index[-1]

        try:
            estado = estado_a_partir_de_operacoes(ops, historico_precos=precos)
        except (ValueError, TypeError, KeyError) as e:
            avisos.append(f"{ticker}: operações inválidas ({e}).")
            continue
        if estado["quantidade"] <= 0:
            continue  # posição encerrada: não aparece

        try:
            dividendos = _sem_fuso(fonte.baixar_dividendos(ticker))
        except Exception:
            dividendos = pd.Series(dtype=float)

        decisao = decidir_posicao(preco, estado)
        if decisao["acao"] == "comprar":
            tipo = tipos.get(ticker)
            if tipo is None:
                avisos.append(f"{ticker}: tipo de ativo não confirmado — compra sugerida sem checar "
                              f"fundamentos.")
            else:
                permitido, motivos = filtro_fundamentos_compra(
                    tipo, fundamentos.get(ticker) or {}, dividendos, data_ref)
                if not permitido:
                    decisao = decidir_posicao(preco, estado, compra_permitida=False,
                                              motivos_bloqueio=motivos)

        moeda = "R$" if ticker.endswith(".SA") else "US$"
        card = montar_card_posicao(ticker, estado, preco, decisao,
                                   quantidade_para_acao(estado, decisao),
                                   renda_12m_por_cota(dividendos, data_ref), moeda)
        card["ticker"] = ticker
        # Campos numéricos para o card visual do painel (a regra só usa texto)
        card["moeda"] = moeda
        card["preco_medio"] = estado["preco_medio"]
        card["preco_atual"] = preco
        card["variacao"] = decisao.get("variacao")
        card["fracao"] = decisao.get("fracao", 0.0)
        card["serie_recente"] = [float(v) for v in precos.iloc[-126:].values]  # ~6 meses
        card["quantidade"] = estado["quantidade"]
        card["tipo"] = tipos.get(ticker)
        card["mercado"] = "B3" if ticker.endswith(".SA") else "EUA"
        card["grupo"] = _TIPO_PARA_GRUPO.get(tipos.get(ticker), "Outros")
        cards.append(card)

    cards.sort(key=chave_ordem_card)
    return cards, avisos
