"""Testes do contexto dos cards: régua de faixas, faixa de 12 meses,
indicadores por tipo de ativo e preço atrativo de hoje (FIIs)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from carteira_analise.contexto import (
    ATENCAO,
    BOM,
    NEUTRO,
    contexto_do_ativo,
    eh_fii_de_tijolo,
    faixa_12m,
    indicadores_contexto,
    preco_atrativo_hoje,
    regua_faixas,
)


# ---------------------------------------------------------------- régua

def test_regua_fora_das_faixas_limites_e_posicao():
    r = regua_faixas(90.96, 94.13)
    assert r["situacao"] == "fora"
    assert r["preco_queda"] == pytest.approx(90.96 * 0.85)
    assert r["preco_realizacao"] == pytest.approx(90.96 * 1.25)
    assert r["degrau"] is None and r["proximo"] is None
    assert 0.25 < r["posicao"] < 0.75  # dentro do trecho do meio


def test_regua_realizacao_degrau_e_proxima_venda_total():
    r = regua_faixas(9.52, 16.22)  # +70%
    assert r["situacao"] == "realizacao"
    assert r["degrau"] == pytest.approx(0.60)
    assert r["proximo"]["faixa"] == pytest.approx(1.00)
    assert r["proximo"]["preco"] == pytest.approx(19.04)
    assert r["proximo"]["venda_total"] is True
    assert r["posicao"] > 0.75


def test_regua_realizacao_ultima_faixa_sem_proxima():
    r = regua_faixas(10.0, 21.0)
    assert r["degrau"] == pytest.approx(1.00)
    assert r["proximo"] is None
    assert r["posicao"] == pytest.approx(0.98)  # marcador limitado à borda


def test_regua_queda_degraus():
    r = regua_faixas(10.0, 8.0)  # −20%
    assert r["situacao"] == "queda"
    assert r["degrau"] == pytest.approx(-0.15)
    assert r["proximo"]["faixa"] == pytest.approx(-0.25)
    assert r["proximo"]["preco"] == pytest.approx(7.5)
    assert r["posicao"] < 0.25
    funda = regua_faixas(10.0, 7.0)  # −30%
    assert funda["degrau"] == pytest.approx(-0.25) and funda["proximo"] is None


def test_regua_nos_limites_exatos():
    assert regua_faixas(10.0, 12.5)["situacao"] == "realizacao"
    assert regua_faixas(10.0, 8.5)["situacao"] == "queda"


def test_regua_sem_dados_validos():
    assert regua_faixas(0, 10) is None
    assert regua_faixas(10, None) is None


# ---------------------------------------------------------------- faixa de 12 meses

def _serie(valores, fim="2026-10-15"):
    idx = pd.bdate_range(end=fim, periods=len(valores))
    return pd.Series(valores, index=idx, dtype=float)


def test_faixa_12m_minima_maxima_e_posicao():
    precos = _serie(list(np.linspace(50, 100, 300)) + [75.0])
    f = faixa_12m(precos)
    janela = precos[precos.index > precos.index[-1] - pd.Timedelta(days=365)]
    assert f["minima"] == pytest.approx(janela.min())
    assert f["maxima"] == pytest.approx(100.0)
    assert f["abaixo_da_maxima"] == pytest.approx(0.25)
    assert f["acima_da_minima"] == pytest.approx(75 / janela.min() - 1)
    assert 0 < f["posicao"] < 1


def test_faixa_12m_ignora_precos_mais_antigos_que_um_ano():
    precos = _serie([200.0] * 100 + [10.0] * 260 + [12.0])
    f = faixa_12m(precos)
    assert f["maxima"] == pytest.approx(12.0)  # o pico de 200 ficou fora da janela


def test_faixa_12m_vazia():
    assert faixa_12m(pd.Series(dtype=float)) is None


# ---------------------------------------------------------------- preço atrativo

def _fii(yield_antigo, yield_recente, anos=4, rend_recente=None):
    """FII com preço constante e rendimento mensal que muda no último ano."""
    idx = pd.bdate_range(end="2026-09-30", periods=252 * anos)
    precos = pd.Series(100.0, index=idx)
    meses = pd.date_range(end="2026-09-15", periods=12 * anos, freq="MS") + pd.Timedelta(days=14)
    valores = [yield_antigo * 100 / 12] * (12 * (anos - 1)) + [(rend_recente or yield_recente) * 100 / 12] * 12
    return precos, pd.Series(valores, index=meses)


def test_preco_atrativo_nao_atende_quando_yield_igual_a_media():
    precos, divs = _fii(0.10, 0.10)
    pa = preco_atrativo_hoje(precos, divs)
    assert pa is not None
    assert pa["yield"] == pytest.approx(0.10, rel=0.02)
    assert pa["limiar"] == pytest.approx(1.15 * pa["yield_medio_24m"])
    assert pa["atende"] is False


def test_preco_atrativo_atende_com_preco_em_queda_e_renda_estavel():
    idx = pd.bdate_range(end="2026-09-30", periods=252 * 4)
    # preço cai 25% no último ano, rendimentos iguais: o yield sobe sem corte de renda
    precos = pd.Series(np.where(idx > idx[-1] - pd.Timedelta(days=200), 75.0, 100.0), index=idx)
    meses = pd.date_range(end="2026-09-15", periods=48, freq="MS") + pd.Timedelta(days=14)
    divs = pd.Series(10 / 12, index=meses)
    pa = preco_atrativo_hoje(precos, divs)
    assert pa["estavel"] is True
    assert pa["yield"] >= pa["limiar"]
    assert pa["atende"] is True


def test_preco_atrativo_sem_historico_suficiente():
    precos, divs = _fii(0.10, 0.10, anos=1)
    assert preco_atrativo_hoje(precos, divs) is None


# ---------------------------------------------------------------- indicadores

def _tec(preco=10.0, sma200=9.0, rsi=50.0):
    return SimpleNamespace(preco_atual=preco, sma200=sma200, rsi14=rsi, historico=None)


def _div(dy=0.08, cresc=0.0):
    return SimpleNamespace(dy_12m_real=dy, crescimento_yoy=cresc)


def test_indicadores_acao_b3_limiares_do_motor():
    itens = indicadores_contexto("acao", _tec(rsi=75), {"pl": 30.0}, _div(dy=0.07))
    nomes = [i["nome"] for i in itens]
    assert nomes == ["Tendência (MM200)", "RSI 14", "P/L", "Dividend yield 12m"]
    por_nome = {i["nome"]: i for i in itens}
    assert por_nome["Tendência (MM200)"]["nivel"] == BOM
    assert por_nome["RSI 14"]["nivel"] == ATENCAO
    assert por_nome["P/L"]["nivel"] == ATENCAO  # acima de 25 na B3
    assert por_nome["Dividend yield 12m"]["nivel"] == BOM  # acima de 6%


def test_indicadores_acao_eua_usam_limiar_proprio_de_pl():
    itens = indicadores_contexto("acao_us", _tec(), {"pl": 30.0}, _div(dy=0.01))
    pl = next(i for i in itens if i["nome"] == "P/L")
    assert pl["nivel"] == NEUTRO  # entre 25 e 40 nos EUA


def test_indicadores_etf_sem_pl():
    itens = indicadores_contexto("etf_br", _tec(), {}, _div(dy=0.05))
    assert [i["nome"] for i in itens] == ["Tendência (MM200)", "RSI 14", "Dividend yield 12m"]


def test_indicadores_fii_sem_tendencia_nem_rsi_e_vacancia_so_no_tijolo():
    pa = {"yield": 0.096, "yield_medio_24m": 0.114, "limiar": 0.131, "estavel": False, "atende": False}
    tijolo = indicadores_contexto("fii", _tec(), {"pvp": 1.2, "qtd_imoveis": 10, "vacancia_media": 0.03},
                                  _div(), pa)
    nomes = [i["nome"] for i in tijolo]
    assert "Tendência (MM200)" not in nomes and "RSI 14" not in nomes
    assert nomes == ["Yield 12 meses", "P/VP", "Rendimentos", "Vacância"]
    por_nome = {i["nome"]: i for i in tijolo}
    assert por_nome["P/VP"]["nivel"] == ATENCAO  # ágio acima de 1,10
    assert por_nome["Rendimentos"]["valor"] == "em queda"
    assert por_nome["Vacância"]["nivel"] == BOM
    papel = indicadores_contexto("fii", _tec(), {"pvp": 0.9, "qtd_imoveis": 0}, _div(), pa)
    assert [i["nome"] for i in papel] == ["Yield 12 meses", "P/VP", "Rendimentos"]


def test_fii_sem_preco_atrativo_usa_yield_e_crescimento_do_historico():
    itens = indicadores_contexto("fii", _tec(), {}, _div(dy=0.09, cresc=-0.2))
    por_nome = {i["nome"]: i for i in itens}
    assert por_nome["Yield 12 meses"]["valor"] == "9,0%"
    assert por_nome["Rendimentos"]["valor"] == "em queda"


def test_eh_fii_de_tijolo():
    assert eh_fii_de_tijolo({"qtd_imoveis": 3})
    assert not eh_fii_de_tijolo({"qtd_imoveis": 0})
    assert not eh_fii_de_tijolo({})


def test_contexto_do_ativo_junta_leitura_indicadores_e_detalhes():
    res = SimpleNamespace(tipo="acao", tec=_tec(), fund={"pl": 10.0}, div=_div(),
                          leitura_combinada_texto="🟡 Sem sinal claro — monitorar",
                          detalhes_timing=["+1 preço acima da MM200"], detalhes_renda=["+1 yield"])
    ctx = contexto_do_ativo(res)
    assert ctx["leitura"].startswith("🟡")
    assert len(ctx["indicadores"]) == 4
    assert ctx["detalhes_timing"] == ["+1 preço acima da MM200"]
    assert ctx["preco_atrativo"] is None
    assert contexto_do_ativo(None) is None


def test_contexto_do_fii_calcula_preco_atrativo_com_os_rendimentos_dados():
    precos, divs = _fii(0.10, 0.10)
    res = SimpleNamespace(tipo="fii", tec=_tec(), fund={}, div=_div(dy=0.10), leitura_combinada_texto="x",
                          detalhes_timing=[], detalhes_renda=[])
    ctx = contexto_do_ativo(res, precos, divs)
    assert ctx["preco_atrativo"] is not None
    assert ctx["indicadores"][0]["nome"] == "Yield 12 meses"


# ---------------------------------------------------------------- yield x CDI líquido

def test_dividend_yield_comparado_ao_cdi_liquido():
    cdi = 0.117
    itens = {i["nome"]: i for i in indicadores_contexto("acao", _tec(), {"pl": 10.0}, _div(dy=0.067),
                                                         cdi_liquido_aa=cdi)}
    dy = itens["Dividend yield 12m"]
    assert dy["nivel"] == ATENCAO  # 6,7% < 70% de 11,7%
    assert "CDI líquido (11,7%)" in dy["leitura"]
    alto = indicadores_contexto("acao", _tec(), {}, _div(dy=0.12), cdi_liquido_aa=cdi)[-1]
    assert alto["nivel"] == BOM
    perto = indicadores_contexto("etf_br", _tec(), {}, _div(dy=0.09), cdi_liquido_aa=cdi)[-1]
    assert perto["nivel"] == NEUTRO
    # EUA: sem comparação com o CDI
    us = indicadores_contexto("acao_us", _tec(), {}, _div(dy=0.03), cdi_liquido_aa=cdi)[-1]
    assert "CDI" not in us["leitura"]


def test_yield_de_fii_comparado_ao_cdi_liquido_com_media_no_detalhe():
    pa = {"yield": 0.125, "yield_medio_24m": 0.11, "limiar": 0.13, "estavel": True, "atende": False}
    y = indicadores_contexto("fii", _tec(), {}, _div(), pa, cdi_liquido_aa=0.117)[0]
    assert y["nivel"] == BOM and "média de 2 anos: 11,0%" in y["leitura"] and "acima do CDI líquido" in y["leitura"]


def test_cdi_liquido_12m():
    from carteira_analise.contexto import cdi_liquido_12m
    idx = pd.bdate_range(end="2026-10-29", periods=400)
    serie = pd.Series(0.05, index=idx)  # 0,05% ao dia útil
    janela = serie[serie.index > idx[-1] - pd.Timedelta(days=365)]
    esperado = ((1.0005) ** len(janela) - 1) * (1 - 0.175)
    assert cdi_liquido_12m(serie) == pytest.approx(esperado)
    assert cdi_liquido_12m(None) is None
