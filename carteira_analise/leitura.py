"""Leitura automática de padrões nos preços extremos de um ativo.

Distingue, para o preço atual, se ele está perto da mínima/máxima do
período e, se estiver, se esse extremo é inédito (sem precedente no
período analisado) ou uma revisita a um patamar que o ativo já havia
visitado antes e do qual já se recuperou. Ver Seção 6.5 do manual.
"""
from __future__ import annotations

import pandas as pd

from .tecnicos import IndicadoresTecnicos, agrupar_datas_em_ciclos

DIAS_RECENTE = 90


def leitura_automatica(fechamento_bruto: pd.Series, tec: IndicadoresTecnicos) -> str:
    """Retorna um texto interpretativo sobre o padrão de preço do ativo,
    dado o histórico bruto (antes da limpeza) e os indicadores já
    calculados. Só analisa ciclos do lado (mínima ou máxima) que o preço
    atual de fato está perto — evita leitura confusa/alarmista quando o
    preço está no meio do range."""
    data_mais_recente = fechamento_bruto.index.max()
    ciclos_minimas = agrupar_datas_em_ciclos(list(fechamento_bruto.nsmallest(10).index))
    ciclos_maximas = agrupar_datas_em_ciclos(list(fechamento_bruto.nlargest(10).index))
    dias_desde_ciclo_min = (data_mais_recente - ciclos_minimas[-1][-1]).days
    dias_desde_ciclo_max = (data_mais_recente - ciclos_maximas[-1][-1]).days

    perto_da_minima = tec.dist_minima_pct < 15
    perto_da_maxima = tec.dist_maxima_pct > -5

    if perto_da_minima:
        if len(ciclos_minimas) == 1 and dias_desde_ciclo_min <= DIAS_RECENTE:
            return (
                "O preço está perto da mínima do período, e os menores preços do "
                "histórico estão concentrados nos últimos ~3 meses — queda recente e "
                "aparentemente real, sem precedente parecido dentro do período analisado."
            )
        if len(ciclos_minimas) > 1:
            primeiro, ultimo = ciclos_minimas[0], ciclos_minimas[-1]
            return (
                "O preço está perto da mínima do período, e o ativo já esteve num "
                f"patamar parecido antes: identifiquei {len(ciclos_minimas)} período(s) de "
                "mínima separados por recuperações no meio. O mais antigo foi por volta de "
                f"{primeiro[0].date()}, e o mais recente começou por volta de "
                f"{ultimo[0].date()}. Vale entender se o motivo daquela recuperação "
                "anterior ainda se aplica agora, ou se dessa vez é diferente."
            )
        return (
            "O preço está perto da mínima do período, mas os menores preços do "
            "histórico não seguem um padrão simples de queda contínua nem de ciclos "
            "repetidos claros. Vale conferir as datas manualmente."
        )

    if perto_da_maxima:
        if len(ciclos_maximas) == 1 and dias_desde_ciclo_max <= DIAS_RECENTE:
            return (
                "O preço está perto da máxima do período, e os maiores preços do "
                "histórico estão concentrados nos últimos ~3 meses — alta recente e "
                "aparentemente real, sem precedente parecido dentro do período analisado."
            )
        if len(ciclos_maximas) > 1:
            primeiro, ultimo = ciclos_maximas[0], ciclos_maximas[-1]
            return (
                "O preço está perto da máxima do período, e o ativo já chegou perto "
                f"desse patamar alto antes: identifiquei {len(ciclos_maximas)} período(s) "
                "de máxima separados por quedas no meio. O mais antigo foi por volta de "
                f"{primeiro[0].date()}, e o mais recente começou por volta de "
                f"{ultimo[0].date()}. Vale considerar se esse nível costuma ser um ponto "
                "de resistência pra esse ativo."
            )
        return (
            "O preço está perto da máxima do período, mas os maiores preços do "
            "histórico não seguem um padrão simples de alta contínua nem de ciclos "
            "repetidos claros. Vale conferir as datas manualmente."
        )

    minimo_abs = fechamento_bruto.min()
    data_minimo_abs = fechamento_bruto.idxmin()
    maximo_abs = fechamento_bruto.max()
    data_maximo_abs = fechamento_bruto.idxmax()
    return (
        f"O preço atual (R$ {tec.preco_atual:.2f}) não está perto nem da mínima nem da "
        f"máxima do período — está numa faixa intermediária. A mínima do período foi "
        f"R$ {minimo_abs:.2f} em {data_minimo_abs.date()}, e a máxima foi "
        f"R$ {maximo_abs:.2f} em {data_maximo_abs.date()}. Nada de especial pra verificar aqui."
    )


def alerta_minima_com_rsi_baixo(tec: IndicadoresTecnicos, limite_rsi: float = 35) -> str | None:
    """Retorna um alerta se o ativo estiver simultaneamente perto da mínima
    do período e com RSI baixo — pode ser barganha genuína ou queda
    estrutural ainda em curso; o notebook não distingue as duas coisas."""
    if tec.dist_minima_pct < 15 and tec.rsi14 is not None and tec.rsi14 < limite_rsi:
        return (
            "Atenção: o ativo está próximo da mínima do período E com RSI baixo ao "
            "mesmo tempo. Isso pode ser uma barganha genuína (preço descontado, "
            "prestes a reagir), ou pode ser uma queda estrutural que ainda não "
            "acabou — RSI baixo não garante que o fundo já foi atingido. Vale checar "
            "se há notícias recentes (resultados, dívida, setor) que expliquem a "
            "queda antes de decidir."
        )
    return None
