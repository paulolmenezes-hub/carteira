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


def test_fii_com_preco_atrativo_mostra_so_yield_x_cdi_e_vacancia():
    """Yield x própria média, P/VP e estabilidade vão para as caixinhas do
    "O preço está atrativo?"; os chips ficam com o CDI e a vacância."""
    pa = {"yield": 0.125, "yield_medio_24m": 0.11, "limiar": 0.13, "estavel": True, "atende": False}
    itens = indicadores_contexto("fii", _tec(), {"segmento": "Logística", "vacancia_media": 0.02, "pvp": 0.9},
                                 _div(), pa, cdi_liquido_aa=0.117)
    assert [i["nome"] for i in itens] == ["Yield 12 meses", "Vacância"]
    assert itens[0]["nivel"] == BOM and "acima do CDI líquido" in itens[0]["leitura"]


# ---------------------------------------------------------------- níveis do preço atrativo (etapa D)

def _fii_nivel(queda_preco, pvp_hoje, pvp_antes=1.0, rend_recente=None):
    """FII de 4 anos: preço 100 até 200 dias atrás e depois ``100 x (1 - queda)``;
    VP constante, escolhido para dar ``pvp_hoje`` no fim."""
    idx = pd.bdate_range(end="2026-09-30", periods=252 * 4)
    preco_final = 100.0 * (1 - queda_preco)
    precos = pd.Series(np.where(idx > idx[-1] - pd.Timedelta(days=200), preco_final, 100.0), index=idx)
    meses = pd.date_range(end="2026-09-15", periods=48, freq="MS") + pd.Timedelta(days=14)
    divs = pd.Series(10 / 12, index=meses)
    if rend_recente is not None:
        divs.iloc[-12:] = rend_recente / 12
    vp_meses = pd.date_range(end="2026-09-01", periods=48, freq="MS")
    vp = pd.Series(preco_final / pvp_hoje, index=vp_meses)
    vp.iloc[:-8] = 100.0 / pvp_antes
    return precos, divs, vp


def test_nivel_forte_com_yield_alto_e_desconto():
    precos, divs, vp = _fii_nivel(0.25, pvp_hoje=0.80)
    pa = preco_atrativo_hoje(precos, divs, vp)
    assert pa["nivel"] == "forte" and pa["pvp"] == pytest.approx(0.80)


def test_nivel_sim_com_yield_alto_sem_desconto():
    precos, divs, vp = _fii_nivel(0.25, pvp_hoje=1.02, pvp_antes=1.02)
    assert preco_atrativo_hoje(precos, divs, vp)["nivel"] == "sim"


def test_nivel_desconto_sem_yield_alto():
    precos, divs, vp = _fii_nivel(0.05, pvp_hoje=0.90)
    pa = preco_atrativo_hoje(precos, divs, vp)
    assert pa["nivel"] == "desconto" and pa["desconto"] and not pa["yield_alto"]


def test_desconto_com_rendimentos_em_queda_e_nao():
    precos, divs, vp = _fii_nivel(0.05, pvp_hoje=0.85, rend_recente=8.0)
    pa = preco_atrativo_hoje(precos, divs, vp)
    assert pa["estavel"] is False and pa["nivel"] == "nao"


def test_sem_vp_da_cvm_usa_pvp_do_fundamentus():
    precos, divs, _ = _fii_nivel(0.05, pvp_hoje=0.9)
    pa = preco_atrativo_hoje(precos, divs, None, pvp_atual=0.88)
    assert pa["pvp"] == pytest.approx(0.88) and pa["pvp_medio_24m"] is None and pa["nivel"] == "desconto"


def test_cdi_liquido_12m():
    from carteira_analise.contexto import cdi_liquido_12m
    idx = pd.bdate_range(end="2026-10-29", periods=400)
    serie = pd.Series(0.05, index=idx)  # 0,05% ao dia útil
    janela = serie[serie.index > idx[-1] - pd.Timedelta(days=365)]
    esperado = ((1.0005) ** len(janela) - 1) * (1 - 0.175)
    assert cdi_liquido_12m(serie) == pytest.approx(esperado)
    assert cdi_liquido_12m(None) is None


def test_fii_de_recebiveis_com_imovel_nao_mostra_vacancia():
    """Caso RECR11: segmento Títulos e Val. Mob., 1 imóvel e vacância de 100%."""
    fund = {"segmento": "Títulos e Val. Mob.", "qtd_imoveis": 1, "vacancia_media": 1.0, "pvp": 0.8}
    nomes = [i["nome"] for i in indicadores_contexto("fii", _tec(), fund, _div())]
    assert "Vacância" not in nomes
    hib = indicadores_contexto("fii", _tec(), {"segmento": "Híbrido", "qtd_imoveis": 4, "vacancia_media": 0.02},
                               _div())
    assert hib[-1]["nome"] == "Vacância (híbrido)"


def test_motivo_sem_historico_aponta_a_causa():
    import pandas as pd
    from carteira_analise.contexto import motivo_sem_historico
    p = pd.Series(100.0, index=pd.bdate_range("2023-06-01", "2026-11-03"))
    d = pd.Series(1.0, index=pd.date_range("2024-06-01", "2026-10-01", freq="MS") + pd.Timedelta(days=14))
    assert "começa em 06/2024" in motivo_sem_historico(p, d)        # rendimentos há menos de 3 anos
    longo = pd.Series(100.0, index=pd.bdate_range("2020-01-01", "2026-11-03"))
    d2 = pd.Series(1.0, index=pd.date_range("2020-01-15", "2026-10-15", freq="MS") + pd.Timedelta(days=14))
    d2 = d2[~d2.index.to_period("M").isin([pd.Period("2024-03"), pd.Period("2024-05")])]
    m = motivo_sem_historico(longo, d2)
    assert "Faltam rendimentos em 2 dos últimos 36 meses (03/2024, 05/2024)" in m
    assert motivo_sem_historico(longo, pd.Series(dtype=float)).startswith("Sem rendimentos")
