"""Indicadores técnicos e tratamento de qualidade de dados de preço.

Este módulo concentra toda a lógica que não depende de nenhuma fonte de
dados específica: recebe uma série de preços (pandas.Series indexada por
data) e devolve indicadores derivados dela. A busca dos dados brutos (hoje
via Yahoo Finance) fica em ``fontes/yahoo.py``, mantendo esta camada
testável com dados sintéticos, sem rede.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def calcular_rsi(precos: pd.Series, janela: int = 14) -> pd.Series:
    """Índice de Força Relativa (RSI) de N dias."""
    delta = precos.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.rolling(janela).mean()
    media_perda = perda.rolling(janela).mean()
    rs = media_ganho / media_perda.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def agrupar_datas_em_ciclos(datas: list, limite_dias: int = 45) -> list[list]:
    """Agrupa datas em 'ciclos' — datas a menos de `limite_dias` de intervalo
    ficam no mesmo grupo; um intervalo maior indica que o preço se afastou e
    voltou depois (um novo ciclo). Usado para diferenciar um ativo fazendo
    mínima/máxima inédita de um ativo revisitando um patamar já visitado.
    """
    if not datas:
        return []
    datas_ordenadas = sorted(datas)
    ciclos = [[datas_ordenadas[0]]]
    for d in datas_ordenadas[1:]:
        if (d - ciclos[-1][-1]).days > limite_dias:
            ciclos.append([d])
        else:
            ciclos[-1].append(d)
    return ciclos


def limpar_outliers_precos(
    fechamento: pd.Series,
    janela: int = 21,
    limite: float = 0.35,
    iteracoes: int = 3,
    janela_salto: int = 10,
    fator_salto: float = 3.0,
    protecao_final: int | None = None,
) -> tuple[pd.Series, int, int]:
    """Corrige preços claramente errados em duas etapas, NESTA ORDEM:

    1) Ticks isolados: comparação com a mediana de uma janela local, em
       várias passadas. Roda PRIMEIRO de propósito — corrige por
       interpolação os erros pontuais de poucos dias antes de qualquer
       outra análise. Isso importa porque, se um tick isolado no MEIO do
       histórico for avaliado antes por um critério global (etapa 2), ele
       pode ser confundido com o início de um bloco inteiro fora de escala,
       levando a truncar (descartar) uma quantidade enorme de dados bons só
       por causa de um erro de poucos dias.

    2) Descontinuidade de patamar (ex.: agrupamento de cotas não ajustado
       retroativamente): SÓ DEPOIS de (1), procura o ponto em que o preço
       muda de escala de um dia para o outro (mediana de 10 pregões antes x
       depois, 3 vezes ou mais) e DESCARTA o trecho anterior a ele.
       Quedas e altas reais, que são graduais, são mantidas — descartá-las
       apagaria a história dos ativos que mais caíram (viés de
       sobrevivência).

    Os últimos ``protecao_final`` pregões (padrão: metade da janela) nunca
    são corrigidos na etapa 1: no fim da série não há pregões seguintes para
    mostrar se um preço muito diferente é erro ou o novo patamar, e trocar o
    preço mais recente pelo anterior escondia quedas reais — caso MRVE3 em
    26/10/2026, queda de cerca de 50% após o 2º turno eleitoral, que o card
    mostrava como R$ 8,23 em vez de R$ 4,07. Movimentos assim são sinalizados
    por ``movimento_brusco_recente``.

    Retorna (serie_limpa, total_pontos_corrigidos, dias_truncados).
    """
    if protecao_final is None:
        protecao_final = janela // 2
    serie = fechamento.copy()

    total_corrigidos = 0
    for _ in range(iteracoes):
        mediana_movel = serie.rolling(janela, center=True, min_periods=1).median()
        desvio_relativo = (serie - mediana_movel).abs() / mediana_movel
        suspeitos = desvio_relativo > limite
        if protecao_final > 0:
            suspeitos.iloc[-protecao_final:] = False  # pregões recentes: preço real, nunca "corrigido"
        n_suspeitos = int(suspeitos.sum())
        if n_suspeitos == 0:
            break
        serie[suspeitos] = np.nan
        serie = serie.interpolate().ffill().bfill()
        total_corrigidos += n_suspeitos

    # 2) Descontinuidade de patamar (grupamento/desdobramento de cotas não
    #    ajustado pelo provedor): o preço muda de escala de um dia para o outro
    #    e PERMANECE no novo patamar. É detectada comparando a mediana dos
    #    `janela_salto` pregões antes e depois de cada ponto: uma mudança de
    #    `fator_salto` vezes ou mais marca o corte, e o trecho anterior é
    #    descartado. Quedas e altas REAIS, mesmo grandes, acontecem ao longo de
    #    semanas ou meses e são mantidas. A versão anterior comparava todo o
    #    histórico com o preço atual e descartava qualquer trecho 60% acima ou
    #    abaixo dele — o que apagava justamente a história dos ativos que mais
    #    caíram (viés de sobrevivência no backtest).
    n_truncados = 0
    k = janela_salto
    if len(serie) >= 2 * k:
        antes = serie.rolling(k).median().shift(1)
        depois = serie[::-1].rolling(k).median()[::-1]
        razao = np.maximum(antes / depois, depois / antes)
        saltos = np.where((razao >= fator_salto).values)[0]
        if len(saltos):
            corte = int(saltos.max())
            alvo = float(depois.iloc[corte])
            # avança até o primeiro ponto já no novo patamar (a etapa 1 pode ter
            # suavizado a fronteira em alguns pregões)
            while corte < len(serie) - 1 and abs(serie.iloc[corte] - alvo) / alvo > limite:
                corte += 1
            if 0 < corte < len(serie) - 1:
                n_truncados = corte
                serie = serie.iloc[corte:]
                total_corrigidos += n_truncados

    return serie, total_corrigidos, n_truncados


def movimento_brusco_recente(precos: pd.Series, dias: int = 10, limite: float = 0.30) -> dict | None:
    """Variação forte e rápida no fim da série: compara o preço atual com a
    mediana dos pregões de 10 a 30 dias antes do trecho recente. Devolve
    {"variacao", "referencia", "preco"} se a variação passar de ``limite``
    (para cima ou para baixo), ou None. Serve para avisar o usuário, porque
    pode ser movimento de mercado ou um evento societário ainda não ajustado
    pela fonte."""
    p = precos.dropna() if precos is not None else pd.Series(dtype=float)
    if len(p) < dias + 20:
        return None
    referencia = float(p.iloc[-(dias + 20):-dias].median())
    preco = float(p.iloc[-1])
    if referencia <= 0:
        return None
    variacao = preco / referencia - 1
    if abs(variacao) < limite:
        return None
    return {"variacao": variacao, "referencia": referencia, "preco": preco}


@dataclass
class IndicadoresTecnicos:
    """Resultado do cálculo de indicadores técnicos para um ativo."""

    preco_atual: float
    sma50: float | None
    sma200: float | None
    rsi14: float | None
    dist_maxima_pct: float
    dist_minima_pct: float
    n_outliers_corrigidos: int
    n_dias_truncados: int
    dias_usados: int
    historico: pd.Series = field(repr=False)


def calcular_indicadores_tecnicos(fechamento_bruto: pd.Series) -> IndicadoresTecnicos | None:
    """Calcula os indicadores técnicos a partir de uma série de preços de
    fechamento já baixada (não faz I/O). Espera um pandas.Series com index
    de datas, ordenado cronologicamente, sem NaN.

    Retorna ``None`` se não houver dados suficientes (< 30 dias, antes ou
    depois da limpeza de outliers).
    """
    fechamento = fechamento_bruto.dropna()
    if len(fechamento) < 30:
        return None

    fechamento, n_outliers, n_truncados = limpar_outliers_precos(fechamento)
    if len(fechamento) < 30:
        return None

    preco_atual = float(fechamento.iloc[-1])
    sma50 = fechamento.rolling(50).mean().iloc[-1] if len(fechamento) >= 50 else np.nan
    sma200 = fechamento.rolling(200).mean().iloc[-1] if len(fechamento) >= 200 else np.nan
    rsi14 = calcular_rsi(fechamento, 14).iloc[-1]

    # Percentil 5%/95%, não mínimo/máximo absolutos: resistente tanto a
    # outliers residuais quanto a mudanças de nível já tratadas acima.
    maxima_periodo = fechamento.quantile(0.95)
    minima_periodo = fechamento.quantile(0.05)
    dist_maxima = (preco_atual - maxima_periodo) / maxima_periodo * 100
    dist_minima = (preco_atual - minima_periodo) / minima_periodo * 100

    return IndicadoresTecnicos(
        preco_atual=preco_atual,
        sma50=float(sma50) if not pd.isna(sma50) else None,
        sma200=float(sma200) if not pd.isna(sma200) else None,
        rsi14=float(rsi14) if not pd.isna(rsi14) else None,
        dist_maxima_pct=float(dist_maxima),
        dist_minima_pct=float(dist_minima),
        n_outliers_corrigidos=n_outliers,
        n_dias_truncados=n_truncados,
        dias_usados=len(fechamento),
        historico=fechamento,
    )
