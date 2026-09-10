"""Backtesting retroativo do motor de pontuação (nota metodológica da Seção
3.1 do Relatório do Projeto Final de Curso — trazido do Horizonte 3 do
roadmap de evolução para o escopo atual do projeto).

A ideia central: reaproveitar os módulos já testados (`tecnicos.py`,
`pontuacao.py`) aplicados retroativamente sobre janelas históricas de preço
— "congelando" uma data de corte no passado e calculando os indicadores
usando só o que era conhecido até ali, sem espiar o futuro (look-ahead
bias) — e comparar o sinal que o motor teria dado naquele momento com o
que de fato aconteceu depois.

Duas métricas, cada uma na dimensão certa (ver nota metodológica da Seção
3.1 do Relatório):

- **F1-score** (macro, 3 classes: compra/manter/venda) para o sinal de
  timing, tratado como um problema de classificação;
- **MAE e R²** para a previsão de renda, tratada como um problema de
  regressão: usa o yield efetivo dos últimos 12 meses na data de corte como
  previsão "ingênua" (persistência) do yield que será pago nos 12 meses
  seguintes, e mede o quão perto essa previsão chega da realidade.

Limitação documentada por transparência: o Yahoo Finance não fornece
fundamentos históricos (P/L, P/VP de datas passadas), então o backtest usa
apenas o componente TÉCNICO do motor de pontuação (tendência, RSI,
distância do período) — não o motor completo usado no dia a dia da
ferramenta, que também pondera fundamentos e renda. O resultado do
backtest deve ser lido como uma validação parcial, do componente técnico
isoladamente, não do motor inteiro.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .tecnicos import IndicadoresTecnicos, calcular_indicadores_tecnicos

HORIZONTE_PADRAO_DIAS = 90
LIMIAR_RETORNO_PADRAO = 0.05  # 5% — parâmetro ajustável, ver docstring do módulo
JANELA_MINIMA_DIAS = 200      # mesmo mínimo usado para a MM200 em tecnicos.py
PASSO_PADRAO_DIAS = 30        # intervalo entre datas de corte sucessivas
LIMIAR_COMPRA_PADRAO = 3      # pontuação mínima pra classificar como "compra"
LIMIAR_VENDA_PADRAO = -2      # pontuação máxima pra classificar como "venda"

CLASSES = ("compra", "manter", "venda")


def _classe_a_partir_de_pontos(
    pontos_tecnicos: int,
    limiar_compra: int = LIMIAR_COMPRA_PADRAO,
    limiar_venda: int = LIMIAR_VENDA_PADRAO,
) -> str:
    """Mesma régua de classificar() em pontuacao.py, simplificada pra 3
    classes (as duas variantes de 'neutro' de classificar() viram 'manter').
    Limiares parametrizáveis — ver Seção 4.3.1 do Relatório (achado do
    backtest sobre a carteira real: recall de 1% pra compra/venda com os
    limiares padrão, motivando o teste de limiares alternativos)."""
    if pontos_tecnicos >= limiar_compra:
        return "compra"
    if pontos_tecnicos <= limiar_venda:
        return "venda"
    return "manter"


def _pontuar_tecnico(tec: IndicadoresTecnicos) -> int:
    """Componente técnico do motor de pontuação, isolado do fundamentalista
    — é o único componente que dá pra reconstruir retroativamente, já que o
    Yahoo Finance não fornece fundamentos históricos. Replica exatamente as
    mesmas regras já usadas em pontuar_acao/pontuar_fii/pontuar_etf para a
    parte técnica (MM200, RSI, distância do período)."""
    pontos = 0
    if tec.sma200 is not None:
        pontos += 1 if tec.preco_atual > tec.sma200 else -1
    if tec.rsi14 is not None:
        if tec.rsi14 < 40:
            pontos += 1
        elif tec.rsi14 > 70:
            pontos -= 1
    if tec.dist_minima_pct < 15:
        pontos += 1
    if tec.dist_maxima_pct > -5:
        pontos -= 1
    return pontos


def rotular_retorno_futuro(
    fechamento: pd.Series,
    data_corte: pd.Timestamp,
    horizonte_dias: int = HORIZONTE_PADRAO_DIAS,
    limiar: float = LIMIAR_RETORNO_PADRAO,
) -> str | None:
    """'Gabarito' retroativo: compara o preço na data de corte com o preço
    ~horizonte_dias depois, e rotula qual sinal teria sido o correto, em
    retrospecto. Retorna None se não houver dado suficiente no futuro (data
    de corte perto demais do fim da série disponível)."""
    pos_corte = fechamento.index.searchsorted(data_corte)
    if pos_corte >= len(fechamento):
        return None
    preco_corte = float(fechamento.iloc[pos_corte])

    data_alvo = data_corte + pd.Timedelta(days=horizonte_dias)
    pos_futuro = fechamento.index.searchsorted(data_alvo)
    if pos_futuro >= len(fechamento):
        return None

    preco_futuro = float(fechamento.iloc[pos_futuro])
    retorno = (preco_futuro - preco_corte) / preco_corte

    if retorno > limiar:
        return "compra"
    if retorno < -limiar:
        return "venda"
    return "manter"


def _yield_ultimos_12m(dividendos_ate_data: pd.Series, preco: float) -> float | None:
    if dividendos_ate_data is None or dividendos_ate_data.empty or not preco:
        return None
    hoje = dividendos_ate_data.index.max()
    ult_12m = dividendos_ate_data[dividendos_ate_data.index > hoje - pd.Timedelta(days=365)]
    if ult_12m.empty:
        return None
    return float(ult_12m.sum()) / preco


@dataclass
class PontoBacktest:
    ticker: str
    data_corte: pd.Timestamp
    classe_prevista: str
    classe_real: str
    dy_trailing_previsto: float | None
    dy_futuro_realizado: float | None


def executar_backtest_ticker(
    fechamento_completo: pd.Series,
    ticker: str,
    dividendos: pd.Series | None = None,
    passo_dias: int = PASSO_PADRAO_DIAS,
    horizonte_dias: int = HORIZONTE_PADRAO_DIAS,
    limiar_retorno: float = LIMIAR_RETORNO_PADRAO,
    limiar_compra: int = LIMIAR_COMPRA_PADRAO,
    limiar_venda: int = LIMIAR_VENDA_PADRAO,
) -> list[PontoBacktest]:
    """Roda o backtest retroativo de um único ticker: percorre datas de
    corte a cada `passo_dias`, calcula o sinal técnico só com dado
    disponível até ali, e compara com o que aconteceu `horizonte_dias`
    depois. Reaproveita calcular_indicadores_tecnicos (mesma função usada
    ao vivo pela ferramenta), só que aplicada a um recorte do passado.
    `limiar_compra`/`limiar_venda` permitem testar réguas de classificação
    diferentes da padrão (3/-2) sem alterar o motor de pontuação ao vivo."""
    if dividendos is None:
        dividendos = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    elif dividendos.index.tz is not None:
        dividendos = dividendos.copy()
        dividendos.index = dividendos.index.tz_localize(None)

    registros: list[PontoBacktest] = []
    limite_superior = len(fechamento_completo) - 1
    indices_corte = range(JANELA_MINIMA_DIAS, limite_superior, passo_dias)

    for idx in indices_corte:
        data_corte = fechamento_completo.index[idx]
        fechamento_ate_corte = fechamento_completo.iloc[: idx + 1]

        tec = calcular_indicadores_tecnicos(fechamento_ate_corte)
        if tec is None:
            continue

        classe_real = rotular_retorno_futuro(fechamento_completo, data_corte, horizonte_dias, limiar_retorno)
        if classe_real is None:
            continue  # sem dado futuro suficiente ainda (data de corte recente demais)

        classe_prevista = _classe_a_partir_de_pontos(_pontuar_tecnico(tec), limiar_compra, limiar_venda)

        dividendos_ate_corte = dividendos[dividendos.index <= data_corte]
        dy_trailing = _yield_ultimos_12m(dividendos_ate_corte, tec.preco_atual)

        data_futuro_12m = data_corte + pd.Timedelta(days=365)
        dividendos_futuros = dividendos[(dividendos.index > data_corte) & (dividendos.index <= data_futuro_12m)]
        dy_futuro = float(dividendos_futuros.sum()) / tec.preco_atual if tec.preco_atual else None

        registros.append(PontoBacktest(
            ticker=ticker, data_corte=data_corte,
            classe_prevista=classe_prevista, classe_real=classe_real,
            dy_trailing_previsto=dy_trailing, dy_futuro_realizado=dy_futuro,
        ))

    return registros


@dataclass
class MetricasClasse:
    precisao: float
    recall: float
    f1: float
    n_real: int


@dataclass
class ResultadoF1:
    f1_macro: float
    por_classe: dict[str, MetricasClasse]
    n_total: int


def calcular_f1_macro(registros: list[PontoBacktest]) -> ResultadoF1:
    """F1-score macro (média simples entre as classes, sem ponderar pelo
    tamanho — evita que a classe mais frequente domine a métrica, que é
    exatamente o vício que motivou a sugestão de usar F1 em vez de simples
    acurácia)."""
    por_classe: dict[str, MetricasClasse] = {}
    for c in CLASSES:
        tp = sum(1 for r in registros if r.classe_prevista == c and r.classe_real == c)
        fp = sum(1 for r in registros if r.classe_prevista == c and r.classe_real != c)
        fn = sum(1 for r in registros if r.classe_prevista != c and r.classe_real == c)
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0.0
        por_classe[c] = MetricasClasse(precisao=precisao, recall=recall, f1=f1, n_real=tp + fn)

    f1_macro = sum(m.f1 for m in por_classe.values()) / len(CLASSES) if registros else 0.0
    return ResultadoF1(f1_macro=f1_macro, por_classe=por_classe, n_total=len(registros))


@dataclass
class ResultadoRegressaoRenda:
    mae: float | None
    r2: float | None
    n: int


def calcular_mae_r2_renda(registros: list[PontoBacktest]) -> ResultadoRegressaoRenda:
    """MAE e R² comparando o yield trailing (12m até a data de corte, usado
    como previsão 'ingênua' de persistência) contra o yield realmente pago
    nos 12 meses seguintes à data de corte."""
    pares = [
        (r.dy_trailing_previsto, r.dy_futuro_realizado)
        for r in registros
        if r.dy_trailing_previsto is not None and r.dy_futuro_realizado is not None
    ]
    if not pares:
        return ResultadoRegressaoRenda(mae=None, r2=None, n=0)

    previstos = np.array([p[0] for p in pares])
    reais = np.array([p[1] for p in pares])

    mae = float(np.mean(np.abs(previstos - reais)))

    media_real = reais.mean()
    ss_tot = float(np.sum((reais - media_real) ** 2))
    ss_res = float(np.sum((reais - previstos) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None

    return ResultadoRegressaoRenda(mae=mae, r2=r2, n=len(pares))
