from __future__ import annotations

import math

import pandas as pd

from carteira_analise.fundamentalistas import (
    calcular_historico_dividendos,
    extrair_fundamentalistas_acao,
    extrair_fundamentalistas_fii,
)


def aproximadamente(valor: float, rel: float = 1e-6) -> object:
    """Comparador leve para floats, sem depender de `pytest.approx`
    dentro do próprio helper — mais simples de ler."""

    class _Aprox:
        def __eq__(self, outro):
            return math.isclose(outro, valor, rel_tol=rel)

        def __repr__(self):
            return f"~{valor}"

    return _Aprox()


class TestExtrairFundamentalistas:
    def test_extrai_campos_de_acao(self):
        info = {
            "trailingPE": 8.5,
            "priceToBook": 1.2,
            "returnOnEquity": 0.18,
            "debtToEquity": 0.5,
            "dividendYield": 1224.0,  # campo bruto inconsistente — deve ser IGNORADO
        }
        fund = extrair_fundamentalistas_acao(info)
        assert fund == {"pl": 8.5, "pvp": 1.2, "roe": 0.18, "divida_patrimonio": 0.5}
        assert "dividend_yield" not in fund  # confirma que o campo problemático não é usado

    def test_extrai_campos_de_fii(self):
        info = {"priceToBook": 0.95, "averageVolume": 12345, "dividendYield": 1224.0}
        fund = extrair_fundamentalistas_fii(info)
        assert fund == {"pvp": 0.95, "volume_medio": 12345}

    def test_campos_ausentes_viram_none(self):
        assert extrair_fundamentalistas_acao({}) == {
            "pl": None,
            "pvp": None,
            "roe": None,
            "divida_patrimonio": None,
        }


class TestCalcularHistoricoDividendos:
    def test_dividendos_vazio_retorna_none(self):
        assert calcular_historico_dividendos(pd.Series(dtype=float), 100.0) is None

    def test_yield_calculado_a_partir_de_dividendos_reais(self, dividendos_crescentes):
        # preco_atual escolhido para dar um yield redondo e fácil de conferir
        preco_atual = 8.0  # soma dos últimos 12m = 12 * 0.8 = 9.6 -> yield 120%... vamos conferir
        div = calcular_historico_dividendos(dividendos_crescentes, preco_atual)
        assert div is not None
        assert div.soma_12m == aproximadamente(9.6)
        assert div.dy_12m_real == aproximadamente(9.6 / 8.0)

    def test_crescimento_ano_contra_ano_positivo(self, dividendos_crescentes):
        div = calcular_historico_dividendos(dividendos_crescentes, 100.0)
        assert div is not None
        # 12 * 0.8 vs 12 * 0.5 -> crescimento de 60%
        assert div.crescimento_yoy == aproximadamente(0.6)

    def test_numero_de_pagamentos_conta_apenas_ultimos_12_meses(self, dividendos_crescentes):
        div = calcular_historico_dividendos(dividendos_crescentes, 100.0)
        assert div is not None
        assert div.n_pagamentos_12m == 12

    def test_yield_nao_e_calculado_sem_preco(self, dividendos_crescentes):
        div = calcular_historico_dividendos(dividendos_crescentes, 0.0)
        assert div is not None
        assert div.dy_12m_real is None
