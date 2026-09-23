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

from .planilha import _construir_operacao_da_posicao_inicial, normalizar_ticker
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
            try:
                ops.setdefault(ticker, []).append({
                    "data": pd.Timestamp(row["data"]), "tipo": tipo,
                    "quantidade": float(row["quantidade"]), "preco": float(row["preco"])})
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


def gerar_posicoes(df_resumo, df_operacoes, fonte, tipos: dict[str, str] | None = None,
                   fundamentos: dict[str, dict] | None = None,
                   periodo: str = "5y") -> tuple[list[dict], list[str]]:
    """Aplica a regra de faixas a cada posição ABERTA da carteira. Devolve
    (cards, avisos): um card por ativo (ver ``montar_card_posicao``), com a
    chave extra ``ticker``, ordenados com quem pede ação hoje primeiro."""
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
        cards.append(card)

    cards.sort(key=lambda c: (c["prioridade"], c["ticker"]))
    return cards, avisos
