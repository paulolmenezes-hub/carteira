"""Cálculo de ganho de investimento — duas formas de uso, conforme o nível de
detalhe que o usuário tiver disponível (Seção 3.2.2 do Relatório do Projeto
Final de Curso, "Concepção da Solução").

1) ``calcular_ganho_posicao_aberta``: dado agregado mínimo (quantidade, preço
   médio, data de início) — não exige nenhum histórico de operações, é o
   ponto de entrada de baixa fricção pra qualquer posição já existente.

2) ``avaliar_operacoes``: registro incremental de operações (compra/venda) —
   opcional, começa a valer a partir do momento em que o usuário passa a
   registrar. Não exige reconstrução do passado: para o período anterior ao
   registro, o Nível 1 acima continua sendo a fonte de verdade.

Ambas as funções são puras (sem I/O de rede), consistentes com o resto do
pacote — quem chama já traz o preço atual e o histórico de dividendos
(por exemplo, via ``fontes.yahoo`` e ``fundamentalistas``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd


def _soma_dividendos_desde(dividendos: pd.Series | None, data_inicio) -> float:
    """Soma os proventos pagos a partir de uma data (inclusive), tolerando
    série vazia/None e index com ou sem timezone."""
    if dividendos is None or dividendos.empty:
        return 0.0
    serie = dividendos.copy()
    if serie.index.tz is not None:
        serie.index = serie.index.tz_localize(None)
    corte = pd.Timestamp(data_inicio)
    return float(serie[serie.index >= corte].sum())


# =============================================================================
# 1) Ganho de posição aberta — dado agregado (Nível 1)
# =============================================================================
@dataclass
class GanhoPosicaoAberta:
    quantidade: float
    preco_medio: float
    preco_atual: float
    valor_investido: float
    valor_atual: float
    ganho_preco: float
    ganho_preco_pct: float
    renda_recebida: float
    ganho_total: float
    ganho_total_pct: float


def calcular_ganho_posicao_aberta(
    quantidade: float,
    preco_medio: float,
    data_inicio,
    preco_atual: float,
    dividendos: pd.Series | None = None,
) -> GanhoPosicaoAberta:
    """Calcula o ganho de uma posição ainda aberta, a partir do dado mínimo
    que praticamente qualquer investidor sabe informar de cabeça: quantidade,
    preço médio pago e data em que passou a ter a posição.

    A renda recebida é somada automaticamente a partir do histórico real de
    dividendos do ativo (mesma fonte já usada em ``fundamentalistas.py``),
    não precisa ser informada pelo usuário — é aqui que a decisão de design
    registrada na Seção 3.2.2 do relatório ("proventos capturados
    automaticamente") se concretiza.
    """
    if quantidade <= 0:
        raise ValueError("quantidade deve ser positiva")
    if preco_medio <= 0:
        raise ValueError("preco_medio deve ser positivo")

    valor_investido = quantidade * preco_medio
    valor_atual = quantidade * preco_atual
    ganho_preco = valor_atual - valor_investido
    ganho_preco_pct = ganho_preco / valor_investido

    renda_recebida = _soma_dividendos_desde(dividendos, data_inicio) * quantidade

    ganho_total = ganho_preco + renda_recebida
    ganho_total_pct = ganho_total / valor_investido

    return GanhoPosicaoAberta(
        quantidade=quantidade,
        preco_medio=preco_medio,
        preco_atual=preco_atual,
        valor_investido=valor_investido,
        valor_atual=valor_atual,
        ganho_preco=ganho_preco,
        ganho_preco_pct=ganho_preco_pct,
        renda_recebida=renda_recebida,
        ganho_total=ganho_total,
        ganho_total_pct=ganho_total_pct,
    )


# =============================================================================
# 2) Registro incremental de operações e avaliação de posições encerradas
# =============================================================================
@dataclass
class Operacao:
    """Uma operação de compra ou venda. ``tipo`` é 'compra' ou 'venda'."""
    data: date
    tipo: str
    quantidade: float
    preco: float

    def __post_init__(self) -> None:
        if self.tipo not in ("compra", "venda"):
            raise ValueError("tipo deve ser 'compra' ou 'venda'")
        if self.quantidade <= 0:
            raise ValueError("quantidade deve ser positiva")
        if self.preco <= 0:
            raise ValueError("preco deve ser positivo")


@dataclass
class ResultadoOperacoes:
    quantidade_aberta: float
    preco_medio_aberto: float | None
    ganho_realizado: float
    ganho_nao_realizado: float | None
    renda_recebida: float
    ganho_total: float
    detalhes: list[str] = field(default_factory=list)


def avaliar_operacoes(
    operacoes: list[Operacao],
    preco_atual: float | None = None,
    dividendos: pd.Series | None = None,
) -> ResultadoOperacoes:
    """Processa uma lista de operações de compra/venda em ordem cronológica,
    usando custo médio ponderado (o mesmo método usado pela Receita Federal
    para apuração de ganho de capital no Brasil) para calcular:

    - o ganho REALIZADO nas vendas já feitas (preço de venda menos o custo
      médio da posição no momento de cada venda);
    - o ganho NÃO REALIZADO do que ainda está em carteira (se `preco_atual`
      for informado);
    - a renda recebida (dividendos) no período coberto pelas operações.

    Limitação conhecida, documentada por transparência: a renda recebida é
    somada por período coberto pelas operações (da primeira à última data
    informada, ou até hoje se a posição seguir aberta), não célula a célula
    por operação — uma simplificação aceitável para o nível de precisão
    desta funcionalidade (Seção 3.2.2 do relatório), mas que pode
    superestimar a renda em carteiras com posição zerada por períodos longos
    entre operações.
    """
    if not operacoes:
        raise ValueError("é necessário informar ao menos uma operação")

    ops_ordenadas = sorted(operacoes, key=lambda o: o.data)

    quantidade_aberta = 0.0
    custo_medio = 0.0
    ganho_realizado = 0.0
    detalhes: list[str] = []

    for op in ops_ordenadas:
        if op.tipo == "compra":
            novo_custo_total = custo_medio * quantidade_aberta + op.preco * op.quantidade
            quantidade_aberta += op.quantidade
            custo_medio = novo_custo_total / quantidade_aberta if quantidade_aberta > 0 else 0.0
            detalhes.append(
                f"{op.data}: compra de {op.quantidade:.0f} a {op.preco:.2f} "
                f"— posição passa a {quantidade_aberta:.0f} (custo médio {custo_medio:.2f})"
            )
        else:  # venda
            if op.quantidade > quantidade_aberta + 1e-9:
                raise ValueError(
                    f"venda de {op.quantidade} em {op.data} excede a posição em aberto "
                    f"({quantidade_aberta}) — confira as operações informadas"
                )
            ganho_da_venda = (op.preco - custo_medio) * op.quantidade
            ganho_realizado += ganho_da_venda
            quantidade_aberta -= op.quantidade
            detalhes.append(
                f"{op.data}: venda de {op.quantidade:.0f} a {op.preco:.2f} "
                f"(custo médio {custo_medio:.2f}) — ganho realizado de {ganho_da_venda:+.2f}"
            )
            if quantidade_aberta <= 1e-9:
                quantidade_aberta = 0.0
                custo_medio = 0.0

    ganho_nao_realizado = None
    if quantidade_aberta > 0 and preco_atual is not None:
        ganho_nao_realizado = (preco_atual - custo_medio) * quantidade_aberta

    data_inicio = ops_ordenadas[0].data
    renda_recebida = _soma_dividendos_desde(dividendos, data_inicio)
    # aproxima pela quantidade média mantida ao longo do período coberto —
    # simplificação documentada na docstring desta função
    quantidades = []
    qtd_acum = 0.0
    for op in ops_ordenadas:
        qtd_acum += op.quantidade if op.tipo == "compra" else -op.quantidade
        quantidades.append(max(qtd_acum, 0.0))
    quantidade_media_periodo = sum(quantidades) / len(quantidades) if quantidades else 0.0
    renda_recebida *= quantidade_media_periodo

    ganho_total = ganho_realizado + (ganho_nao_realizado or 0.0) + renda_recebida

    return ResultadoOperacoes(
        quantidade_aberta=quantidade_aberta,
        preco_medio_aberto=custo_medio if quantidade_aberta > 0 else None,
        ganho_realizado=ganho_realizado,
        ganho_nao_realizado=ganho_nao_realizado,
        renda_recebida=renda_recebida,
        ganho_total=ganho_total,
        detalhes=detalhes,
    )
