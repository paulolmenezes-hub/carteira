from __future__ import annotations

import pytest

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
        div = calcular_historico_dividendos(dividendos_crescentes, preco_atual, data_ref="2025-07-15")
        assert div is not None
        assert div.soma_12m == aproximadamente(9.6)
        assert div.dy_12m_real == aproximadamente(9.6 / 8.0)

    def test_crescimento_ano_contra_ano_positivo(self, dividendos_crescentes):
        div = calcular_historico_dividendos(dividendos_crescentes, 100.0, data_ref="2025-07-15")
        assert div is not None
        # 12 * 0.8 vs 12 * 0.5 -> crescimento de 60%
        assert div.crescimento_yoy == aproximadamente(0.6)

    def test_numero_de_pagamentos_conta_apenas_ultimos_12_meses(self, dividendos_crescentes):
        div = calcular_historico_dividendos(dividendos_crescentes, 100.0, data_ref="2025-07-15")
        assert div is not None
        assert div.n_pagamentos_12m == 12

    def test_ativo_que_parou_de_pagar_nao_mostra_yield_antigo(self, dividendos_crescentes):
        """Caso MRVE3: os 12 meses contam a partir de hoje, não do último
        pagamento — sem proventos recentes, o yield é zero."""
        div = calcular_historico_dividendos(dividendos_crescentes, 4.0, data_ref="2026-10-30")
        assert div.soma_12m == 0 and div.dy_12m_real == 0 and div.n_pagamentos_12m == 0

    def test_yield_nao_e_calculado_sem_preco(self, dividendos_crescentes):
        div = calcular_historico_dividendos(dividendos_crescentes, 0.0)
        assert div is not None
        assert div.dy_12m_real is None


import pytest as _pytest


@_pytest.mark.parametrize("fund, esperado", [
    ({"segmento": "Títulos e Val. Mob.", "qtd_imoveis": 1}, "papel"),   # RECR11: imóvel recebido em pagamento
    ({"segmento": "Logística", "qtd_imoveis": 0}, "tijolo"),
    ({"segmento": "Lajes Corporativas"}, "tijolo"),
    ({"segmento": "Escritórios", "qtd_imoveis": 12}, "tijolo"),     # "cri" dentro de "escritórios" não é CRI
    ({"segmento": "CRI", "qtd_imoveis": 0}, "papel"),
    ({"segmento": "FOF"}, "papel"),
    ({"segmento": "Híbrido", "qtd_imoveis": 5}, "hibrido"),
    ({"segmento": "Outros", "qtd_imoveis": 3}, "tijolo"),              # sem segmento útil: vale a quantidade
    ({"segmento": "", "qtd_imoveis": 0}, "papel"),
    ({}, None),
])
def test_classificar_fii_pelo_segmento(fund, esperado):
    from carteira_analise.fundamentalistas import classificar_fii, tem_imoveis
    assert classificar_fii(fund) == esperado
    assert tem_imoveis(fund) == (esperado in ("tijolo", "hibrido"))


@_pytest.mark.parametrize("fund, esperado", [
    ({"segmento": "Multicategoria", "qtd_imoveis": 1, "imoveis_pl": 0.032}, "papel"),   # RECR11
    ({"segmento": "Multicategoria", "qtd_imoveis": 8, "imoveis_pl": 0.35}, "hibrido"),
    ({"segmento": "Outros", "qtd_imoveis": 12, "imoveis_pl": 0.92}, "tijolo"),
    ({"segmento": "Multicategoria", "qtd_imoveis": 3}, "tijolo"),                       # sem Imóveis/PL
    ({"segmento": "Logística", "imoveis_pl": 0.05}, "papel"),   # desde 05/11/2026: Imóveis/PL confere o segmento
])
def test_segmento_generico_usa_imoveis_sobre_pl(fund, esperado):
    from carteira_analise.fundamentalistas import classificar_fii
    assert classificar_fii(fund) == esperado


def test_imoveis_pl_da_pagina_de_detalhes():
    from carteira_analise.fontes.fundamentus import imoveis_pl_de_html
    html = ('<tr><td class="label w2"><span class="help tips" title="x">?</span><span class="txt">Imóveis/PL do FII'
            '</span></td><td class="data w1"><span class="txt">3,2%</span></td></tr>'
            '<tr><td><span class="txt">Preço do m2</span></td><td><span class="txt">5.109</span></td></tr>')
    assert imoveis_pl_de_html(html) == _pytest.approx(0.032)
    assert imoveis_pl_de_html("<p>sem o campo</p>") is None



@pytest.mark.parametrize("fund, esperado", [
    ({"segmento": "Logística", "qtd_imoveis": 3, "imoveis_pl": 0.05}, "papel"),   # MXRF11: cadastro x carteira
    ({"segmento": "Logística", "qtd_imoveis": 3, "imoveis_pl": 0.35}, "hibrido"),
    ({"segmento": "Logística", "qtd_imoveis": 12, "imoveis_pl": 0.92}, "tijolo"),
    ({"segmento": "Logística", "qtd_imoveis": 12}, "tijolo"),                    # sem Imóveis/PL: vale o segmento
])
def test_segmento_de_tijolo_conferido_pelo_imoveis_pl(fund, esperado):
    from carteira_analise.fundamentalistas import classificar_fii, motivo_classificacao
    assert classificar_fii(fund) == esperado
    motivo = motivo_classificacao(fund)
    assert (motivo is None) == (esperado == "tijolo")
    if motivo:
        assert "Logística" in motivo and "Imóveis/PL" in motivo
