"""Processamento de carteira combinando duas abas (Resumo + Operações) de
uma planilha — mesmo modelo de dados usado na Seção 9.3 do notebook do
protótipo, agora testado como parte do pacote (Seção 3.2.2 do Relatório:
"o usuário faz upload da planilha e vê o resultado").

Modelo de dados: a aba **Resumo** é o ponto de partida (posição inicial de
cada ativo — o "saldo de abertura"); a aba **Operações** traz as
movimentações feitas a partir dali. As duas são combinadas por ativo: a
posição inicial (se existir) entra como a primeira "compra" sintética, e
cada operação lançada depois se soma em cima dela — igual um extrato. Um
ativo pode ter só Resumo, só Operações, ou as duas (mais comum).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .ganhos import Operacao, avaliar_operacoes

_PADRAO_TICKER_B3 = re.compile(r"^[A-Z]{4}\d{1,2}$")


def normalizar_ticker(ticker_bruto: str) -> tuple[str, bool]:
    """Limpa espaços e, se o ticker parecer um código B3 (4 letras + 1-2
    dígitos) sem o sufixo '.SA', completa automaticamente — ex: 'POMO4' ->
    'POMO4.SA'. Devolve (ticker_normalizado, foi_corrigido)."""
    t = str(ticker_bruto).strip().upper()
    if t.endswith(".SA") or "." in t or "-" in t:
        return t, False
    if _PADRAO_TICKER_B3.match(t):
        return t + ".SA", True
    return t, False


def _normalizar_nome_coluna(c: str) -> str:
    c = str(c).strip().lower()
    mapa_acentos = (
        ("á", "a"), ("à", "a"), ("â", "a"), ("ã", "a"),
        ("é", "e"), ("ê", "e"),
        ("í", "i"),
        ("ó", "o"), ("ô", "o"), ("õ", "o"),
        ("ú", "u"), ("ü", "u"),
        ("ç", "c"),
    )
    for de, para in mapa_acentos:
        c = c.replace(de, para)
    return c.replace("_", " ")


def normalizar_aba_resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas esperadas: ticker, quantidade, data_inicio, e preco_medio OU
    valor_investido (pelo menos um dos dois)."""
    mapa = {}
    for col in df.columns:
        c = _normalizar_nome_coluna(col)
        if c in ("ticker", "ativo", "codigo", "papel"):
            mapa[col] = "ticker"
        elif c in ("quantidade", "qtd", "qtde", "quantity"):
            mapa[col] = "quantidade"
        elif c in ("preco medio", "pm", "preco medio r$", "preco", "preco entrada", "preco compra"):
            mapa[col] = "preco_medio"
        elif c in ("valor investido", "valor total", "valor aplicado", "total investido"):
            mapa[col] = "valor_investido"
        elif c in ("data inicio", "data", "data compra", "data aquisicao", "data entrada"):
            mapa[col] = "data_inicio"
    df_novo = df.rename(columns=mapa)
    faltando = [c for c in ("ticker", "quantidade", "data_inicio") if c not in df_novo.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias não encontradas na aba Resumo: {faltando}.")
    if "preco_medio" not in df_novo.columns and "valor_investido" not in df_novo.columns:
        raise ValueError("A aba Resumo precisa ter 'preco_medio' OU 'valor_investido' preenchido.")
    for opcional in ("preco_medio", "valor_investido"):
        if opcional not in df_novo.columns:
            df_novo[opcional] = None
    return df_novo[["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"]]


def normalizar_aba_operacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas esperadas: ticker, tipo (compra/venda), quantidade, preco, data."""
    mapa = {}
    for col in df.columns:
        c = _normalizar_nome_coluna(col)
        if c in ("ticker", "ativo", "codigo", "papel"):
            mapa[col] = "ticker"
        elif c in ("tipo", "operacao", "movimento", "tipo de operacao"):
            mapa[col] = "tipo"
        elif c in ("quantidade", "qtd", "qtde", "quantity"):
            mapa[col] = "quantidade"
        elif c in ("preco", "preco unitario", "valor", "valor unitario"):
            mapa[col] = "preco"
        elif c in ("data", "data operacao", "data da operacao"):
            mapa[col] = "data"
    df_novo = df.rename(columns=mapa)
    faltando = [c for c in ("ticker", "tipo", "quantidade", "preco", "data") if c not in df_novo.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias não encontradas na aba Operações: {faltando}.")
    return df_novo[["ticker", "tipo", "quantidade", "preco", "data"]]


def _construir_operacao_da_posicao_inicial(row) -> Operacao:
    quantidade = float(row["quantidade"])
    preco_medio = row.get("preco_medio")
    if preco_medio is None or (isinstance(preco_medio, float) and pd.isna(preco_medio)):
        valor_investido = row.get("valor_investido")
        if valor_investido is None or (isinstance(valor_investido, float) and pd.isna(valor_investido)):
            raise ValueError("informe preco_medio ou valor_investido na aba Resumo")
        preco_medio = float(valor_investido) / quantidade
    return Operacao(data=pd.Timestamp(row["data_inicio"]).date(), tipo="compra",
                     quantidade=quantidade, preco=float(preco_medio))


@dataclass
class LinhaCarteira:
    ticker: str
    moeda: str
    situacao: str  # 'Aberta', 'Encerrada' ou 'erro'
    origem: str
    avisos: list[str] = field(default_factory=list)
    quantidade_aberta: float | None = None
    preco_medio_atual: float | None = None
    ganho_realizado: float | None = None
    ganho_nao_realizado: float | None = None
    renda_recebida: float | None = None
    ganho_total: float | None = None
    erro: str | None = None


def processar_carteira_combinada(
    df_resumo: pd.DataFrame | None,
    df_operacoes: pd.DataFrame | None,
    fonte,
    periodo: str = "5y",
) -> list[LinhaCarteira]:
    """Uma ``LinhaCarteira`` por ativo, combinando a posição inicial (aba
    Resumo, quando existir) com as movimentações (aba Operações, quando
    existirem). ``fonte`` precisa expor ``baixar_precos(ticker, periodo)``
    e ``baixar_dividendos(ticker)`` (mesmo protocolo de ``fontes.yahoo``,
    usado em ``carteira.analisar_ativo``) — sem I/O de rede direto aqui,
    o que mantém esta função testável com fonte falsa."""
    tickers_resumo: dict[str, object] = {}
    if df_resumo is not None:
        for _, row in df_resumo.iterrows():
            ticker, _ = normalizar_ticker(row["ticker"])
            tickers_resumo[ticker] = row

    tickers_operacoes: dict[str, pd.DataFrame] = {}
    if df_operacoes is not None and not df_operacoes.empty:
        tickers_normalizados = df_operacoes["ticker"].apply(lambda t: normalizar_ticker(t)[0])
        for ticker, grupo in df_operacoes.groupby(tickers_normalizados):
            tickers_operacoes[ticker] = grupo

    todos_tickers = sorted(set(tickers_resumo) | set(tickers_operacoes))
    linhas: list[LinhaCarteira] = []

    for ticker in todos_tickers:
        moeda = "R$" if ticker.endswith(".SA") else "US$"
        operacoes: list[Operacao] = []
        avisos: list[str] = []

        if ticker in tickers_resumo:
            try:
                operacoes.append(_construir_operacao_da_posicao_inicial(tickers_resumo[ticker]))
            except (ValueError, TypeError) as e:
                avisos.append(f"posição inicial (aba Resumo) inválida — {e}")

        n_operacoes_lancadas = 0
        if ticker in tickers_operacoes:
            for _, row in tickers_operacoes[ticker].iterrows():
                tipo_bruto = str(row["tipo"]).strip().lower()
                if tipo_bruto in ("compra", "buy", "c"):
                    tipo = "compra"
                elif tipo_bruto in ("venda", "sell", "v"):
                    tipo = "venda"
                else:
                    avisos.append(f"tipo inválido {row['tipo']!r} numa operação (use compra/venda)")
                    continue
                try:
                    data_op = pd.Timestamp(row["data"]).date()
                    quantidade_op = float(row["quantidade"])
                    preco_op = float(row["preco"])
                except (ValueError, TypeError):
                    avisos.append(f"operação com data/quantidade/preço ilegível (data={row['data']!r})")
                    continue
                operacoes.append(Operacao(data=data_op, tipo=tipo, quantidade=quantidade_op, preco=preco_op))
                n_operacoes_lancadas += 1

        if not operacoes:
            linhas.append(LinhaCarteira(ticker=ticker, moeda=moeda, situacao="erro",
                                         origem="", avisos=avisos,
                                         erro="nenhuma posição/operação válida"))
            continue

        try:
            preco_atual = float(fonte.baixar_precos(ticker, periodo).iloc[-1])
        except Exception:
            preco_atual = None
        try:
            divs = fonte.baixar_dividendos(ticker)
        except Exception:
            divs = pd.Series(dtype=float)

        try:
            r = avaliar_operacoes(operacoes, preco_atual=preco_atual, dividendos=divs)
        except ValueError as e:
            linhas.append(LinhaCarteira(ticker=ticker, moeda=moeda, situacao="erro",
                                         origem="", avisos=avisos, erro=str(e)))
            continue

        situacao = "Aberta" if r.quantidade_aberta > 0 else "Encerrada"
        origem = []
        if ticker in tickers_resumo:
            origem.append("posição inicial")
        if n_operacoes_lancadas:
            origem.append(f"{n_operacoes_lancadas} operação(ões)")

        linhas.append(LinhaCarteira(
            ticker=ticker, moeda=moeda, situacao=situacao, origem=" + ".join(origem), avisos=avisos,
            quantidade_aberta=r.quantidade_aberta, preco_medio_atual=r.preco_medio_aberto,
            ganho_realizado=r.ganho_realizado, ganho_nao_realizado=r.ganho_nao_realizado,
            renda_recebida=r.renda_recebida, ganho_total=r.ganho_total,
        ))

    return linhas


def achar_aba(abas: dict[str, pd.DataFrame], nomes_possiveis: list[str]) -> pd.DataFrame | None:
    """Procura uma aba pelo nome, tolerando maiúsculas/minúsculas e acento."""
    alvo = {_normalizar_nome_coluna(n) for n in nomes_possiveis}
    for nome_real, df in abas.items():
        if _normalizar_nome_coluna(nome_real) in alvo:
            return df
    return None
