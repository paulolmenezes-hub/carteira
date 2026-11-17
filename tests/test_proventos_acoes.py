"""Proventos de ações: lacunas do Yahoo completadas com o Fundamentus (17/11/2026, caso CPFE3)."""
from __future__ import annotations

import pandas as pd
import pytest

from carteira_analise.fontes.rendimentos import FonteComRendimentosConferidos, completar_proventos_acao


def _s(pares):
    return pd.Series([v for _, v in pares], index=pd.to_datetime([d for d, _ in pares]))


def test_completa_anos_sem_registro_no_yahoo():
    # Yahoo (data ex) só tem 2023; Fundamentus (data-com) tem 2023, 2024 e 2025
    yahoo = _s([("2023-05-03", 1.0), ("2023-11-06", 1.2)])
    fund = _s([("2023-05-02", 1.0), ("2023-11-03", 1.2), ("2024-05-02", 1.5), ("2024-11-04", 1.1),
               ("2025-05-02", 1.6)])
    serie, info = completar_proventos_acao(yahoo, fund)
    assert info["adicionados"] == 3
    assert serie.sum() == pytest.approx(1.0 + 1.2 + 1.5 + 1.1 + 1.6)
    assert serie["2023"].sum() == pytest.approx(2.2)  # pagamentos já no Yahoo não são duplicados


def test_sem_lacuna_devolve_o_yahoo():
    yahoo = _s([("2024-05-03", 1.0), ("2024-11-06", 0.5)])
    fund = _s([("2024-05-02", 0.7), ("2024-05-02", 0.3), ("2024-11-05", 0.5)])  # dividendo + JCP na mesma data
    serie, info = completar_proventos_acao(yahoo, fund)
    assert info["adicionados"] == 0 and serie.sum() == pytest.approx(1.5)
    assert completar_proventos_acao(yahoo, None)[1]["adicionados"] == 0


def test_escala_do_fundamentus_sem_ajuste_de_desdobramento():
    # desdobramento de 1 para 2 em 2024: Yahoo ajusta os proventos antigos (metade), Fundamentus não
    eventos = pd.Series([2.0], index=pd.to_datetime(["2024-01-15"]))
    yahoo = _s([("2023-05-03", 1.0), ("2024-05-03", 1.0)])
    fund = _s([("2023-05-02", 2.0), ("2023-11-03", 2.4), ("2024-05-02", 1.0)])
    serie, info = completar_proventos_acao(yahoo, fund, eventos)
    assert info["adicionados"] == 1
    assert serie["2023-11"].sum() == pytest.approx(1.2)  # 2,40 da época ÷ 2


def test_fonte_completa_so_acoes():
    class Base:
        def baixar_dividendos(self, t):
            return _s([("2023-05-03", 1.0)])

        def baixar_eventos(self, t):
            return pd.Series(dtype=float)

    fund = _s([("2023-05-02", 1.0), ("2024-05-02", 1.5)])
    fonte = FonteComRendimentosConferidos(Base(), {"CPFE3.SA": "acao", "IVVB11.SA": "etf_br"},
                                          lambda t: {}, proventos_da_acao=lambda t: fund)
    assert fonte.baixar_dividendos("CPFE3.SA").sum() == pytest.approx(2.5)
    assert fonte.complementos_acoes["CPFE3.SA"]["adicionados"] == 1
    assert fonte.baixar_dividendos("IVVB11.SA").sum() == pytest.approx(1.0)
