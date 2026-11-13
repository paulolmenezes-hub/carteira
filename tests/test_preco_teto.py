"""Preço-teto (Bazin com a régua do painel), teste de 11/2026."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carteira_analise.oportunidades import avaliar_preco_teto_mes_a_mes, preco_teto_hoje, resumir_criterio


def _dados(preco_inicial=10.0, cair=False):
    datas = pd.bdate_range("2015-01-02", "2025-12-31")
    tendencia = np.linspace(0, -0.5 if cair else 0.5, len(datas))
    precos = pd.Series(preco_inicial * np.exp(tendencia), index=datas)
    divs = pd.Series(0.5, index=pd.date_range("2015-03-15", "2025-12-15", freq="6MS"))  # R$ 1,00 por ano
    cdi = pd.Series(np.cumprod(np.full(len(datas), 1 + 0.10 / 252)), index=datas)  # CDI de ~10,5% a.a.
    return precos, divs, cdi


def test_teto_usa_a_regua_de_80_por_cento_do_cdi():
    precos, divs, cdi = _dados()
    df = avaliar_preco_teto_mes_a_mes(precos, divs, cdi, "acao")
    linha = df[df["data"] >= "2022-01-01"].iloc[0]
    t = linha["data"]
    ini = cdi[cdi.index <= t - pd.DateOffset(months=12)].iloc[-1]
    cdi_liq = (cdi[cdi.index <= t].iloc[-1] / ini - 1) * (1 - 0.175)
    assert linha["limite_dy"] == pytest.approx(0.80 * cdi_liq, rel=1e-9)
    assert linha["teto_12m"] == pytest.approx(linha["div12"] / linha["limite_dy"])
    assert linha["div_medio_5a"] == pytest.approx(1.0, rel=0.05)


def test_criterios_e_formato_do_resumo():
    precos, divs, cdi = _dados(preco_inicial=8.0)  # preço baixo: abaixo do teto (~1/0,069 ≈ R$ 14,5) no começo
    df = avaliar_preco_teto_mes_a_mes(precos, divs, cdi, "acao")
    assert set(df["criterio_teto_12m"].dropna().unique()) <= {0.0, 1.0}
    assert df["criterio_teto_5a"].iloc[0] != df["criterio_teto_5a"].iloc[0]  # NaN sem 5 anos de histórico
    assert df["criterio_teto_5a"].dropna().any()
    df["ticker"] = "X"
    r = resumir_criterio(df, "criterio_teto_12m", n_bootstrap=50)
    assert r is not None and r["casos"] > 0
    assert df["retorno_12m"].iloc[-1] != df["retorno_12m"].iloc[-1]  # últimos 12 meses: sem resultado


def test_preco_teto_hoje():
    precos, divs, _ = _dados()
    t = preco_teto_hoje(precos, divs, cdi_liquido_aa=0.10, tipo="acao")
    assert t["limite_dy"] == pytest.approx(0.08)
    assert t["teto_12m"] == pytest.approx(t["div12"] / 0.08)
    assert t["teto_5a"] == pytest.approx(1.0 / 0.08, rel=0.05)
    assert preco_teto_hoje(precos, pd.Series(dtype=float), 0.10) is None


def test_preco_teto_bazin_como_o_mercado_calcula():
    """PSSA3 (13/11/2026): R$ 3,00 por ação em 12 meses ÷ 6% = R$ 50,00."""
    from carteira_analise.oportunidades import preco_teto_bazin
    divs = pd.Series([0.35] * 10 + [1.0, 1.0, 1.0],
                     index=pd.to_datetime(["2021-12-15", "2022-06-15", "2022-12-15", "2023-06-15", "2023-12-15",
                                           "2024-06-15", "2024-12-15", "2025-03-15", "2025-06-15", "2025-09-15",
                                           "2026-01-15", "2026-04-15", "2026-09-15"]))
    t = preco_teto_bazin(divs, 46.61, hoje="2026-11-13", cdi_liquido_aa=0.117)
    assert t["div12"] == pytest.approx(3.0) and t["teto"] == pytest.approx(50.0)
    assert t["teto_regua"] == pytest.approx(3.0 / (0.8 * 0.117))
    assert t["div_medio_5a"] == pytest.approx((0.35 * 10 + 3.0) / 5)
    assert t["dividendos_fora_da_media"] == "acima"  # 3,00 contra média de 1,30: teto inflado
    assert preco_teto_bazin(divs, 46.61, hoje="2030-01-01") is None  # sem proventos em 12 meses
