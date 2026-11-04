import numpy as np
import pandas as pd
import pytest

from carteira_analise.backtest_cenarios import indice_cdi_acumulado
from carteira_analise.oportunidades import avaliar_fii_mes_a_mes, frase_resultado, resumir_criterio

DATAS = pd.bdate_range("2016-01-04", "2025-12-31")
MESES = pd.date_range("2016-01-01", "2025-12-01", freq="MS")


def _fii(preco_fn, div=1.0):
    p = pd.Series([preco_fn(i) for i in range(len(DATAS))], index=DATAS, dtype=float)
    d = pd.Series(div, index=MESES)
    return p, d


def test_colunas_e_media_sem_olhar_o_futuro():
    p, d = _fii(lambda i: 100.0)
    df = avaliar_fii_mes_a_mes(p, d, None)
    assert {"yield", "yield_medio_24m", "criterio_yield", "retorno_12m", "superou_cdi"} <= set(df.columns)
    assert df["yield_medio_24m"].iloc[:MESES.size // 4].isna().any()  # precisa de 24 meses antes
    assert df["retorno_12m"].iloc[-1] != df["retorno_12m"].iloc[-1]  # NaN: sem 12 meses à frente
    assert df["yield"].iloc[-1] == pytest.approx(0.12)
    # nenhum yield calculado com menos de 12 meses de rendimentos (aquecimento)
    assert df["yield"].min() == pytest.approx(0.12)
    # rendimento constante é sempre "estável" (janelas por mês do calendário)
    assert df["estavel"].iloc[-60:].all()


def test_fii_recem_listado_nao_dispara_no_aquecimento():
    p, d = _fii(lambda i: 100.0)
    p, d = p[p.index >= "2020-06-01"], d[d.index >= "2020-06-01"]
    df = avaliar_fii_mes_a_mes(p, d, None)
    assert df["data"].min() >= pd.Timestamp("2021-06-01")
    assert not df["criterio_yield"].eq(1).any()


def test_preco_estavel_nao_dispara_e_retorno_soma_rendimentos():
    p, d = _fii(lambda i: 100.0)
    df = avaliar_fii_mes_a_mes(p, d, None).dropna(subset=["retorno_12m", "yield_medio_24m"])
    assert not df["criterio_yield"].eq(1).any()
    assert df["retorno_12m"].median() == pytest.approx(0.12)


def test_queda_de_preco_com_rendimento_estavel_dispara_caminho_2():
    # cota cai 30% em 2021 (yield sobe) e se recupera depois
    def preco(i):
        ano = DATAS[i].year
        return 100.0 if ano < 2021 else (70.0 if ano == 2021 else 100.0)
    p, d = _fii(preco)
    df = avaliar_fii_mes_a_mes(p, d, None)
    disparos = df[df["criterio_yield"] == True]  # noqa: E712
    assert len(disparos) and disparos["data"].dt.year.eq(2021).all()
    # comprar barato em 2021 bateu o CDI zero
    assert disparos["superou_cdi"].dropna().eq(1).all()


def test_rendimento_em_queda_bloqueia_o_criterio():
    def preco(i):
        return 100.0 if DATAS[i].year < 2021 else 60.0
    p = pd.Series([preco(i) for i in range(len(DATAS))], index=DATAS, dtype=float)
    d = pd.Series([1.0 if m.year < 2021 else 0.5 for m in MESES], index=MESES)  # rendimento caiu 50%
    df = avaliar_fii_mes_a_mes(p, d, None)
    em_2021 = df[df["data"].dt.year == 2021]
    # o yield de 12 meses ainda carrega os pagamentos antigos e parece alto (yield trap),
    # mas a média dos últimos 3 meses mostra o corte: o critério não dispara
    assert em_2021["yield"].max() > 0.15
    assert not em_2021["criterio_yield"].eq(1).any()


def test_caminho_3_com_juro_real_e_cdi():
    p, d = _fii(lambda i: 100.0)
    juro = pd.Series([0.06 if dt.year < 2021 else 0.03 for dt in DATAS], index=DATAS)  # juro real cai 3 p.p.
    cdi = indice_cdi_acumulado(DATAS, None, cdi_aa=0.10)
    df = avaliar_fii_mes_a_mes(p, d, cdi, juro)
    assert df["premio"].iloc[-1] == pytest.approx(0.12 - 0.03)
    assert (df[df["data"].dt.year == 2021]["criterio_premio"] == True).any()  # noqa: E712
    # 10% ao ano em 12 meses, IR de 17,5%
    assert df["cdi_liquido_12m"].dropna().iloc[0] == pytest.approx(0.10 * 0.825, rel=0.01)
    # 12% de renda com preço parado contra CDI líquido de ~8,3%: superou
    assert df["superou_cdi"].dropna().eq(1).all()


def test_historico_curto_e_sem_rendimento():
    p = pd.Series(100.0, index=DATAS[:30])
    assert avaliar_fii_mes_a_mes(p, pd.Series(dtype=float), None).empty
    p2, _ = _fii(lambda i: 100.0)
    assert avaliar_fii_mes_a_mes(p2, pd.Series(dtype=float), None).empty


def test_resumo_bootstrap_e_frase():
    casos = []
    for k, queda in enumerate([True, True, True, False, False]):
        def preco(i, queda=queda):
            return 70.0 if (queda and DATAS[i].year == 2021) else 100.0
        p, d = _fii(preco)
        cdi = indice_cdi_acumulado(DATAS, None, cdi_aa=0.10)
        df = avaliar_fii_mes_a_mes(p, d, cdi)
        df["ticker"] = f"F{k}"
        casos.append(df)
    casos = pd.concat(casos, ignore_index=True)
    r = resumir_criterio(casos, "criterio_yield", n_bootstrap=200)
    assert r["fiis"] == 5 and r["fiis_com_criterio"] == 3 and r["casos_com_criterio"] > 0
    assert r["taxa_com_criterio"] >= r["taxa_base"]
    assert r["ic90_diferenca"] is not None and r["ic90_diferenca"][0] <= r["ic90_diferenca"][1]
    f = frase_resultado(r, "quando o yield estava 15% acima da própria média")
    assert "Em 5 FIIs" in f and "superou o CDI líquido" in f and "Intervalo de confiança" in f


def test_resumo_sem_casos_e_sem_criterio():
    assert resumir_criterio(pd.DataFrame({"ticker": [], "criterio_yield": [], "superou_cdi": [],
                                          "retorno_12m": [], "cdi_liquido_12m": [], "data": []}),
                            "criterio_yield") is None
    assert "sem casos" in frase_resultado(None, "x")
    p, d = _fii(lambda i: 100.0)
    df = avaliar_fii_mes_a_mes(p, d, None); df["ticker"] = "A"
    r = resumir_criterio(df, "criterio_yield", n_bootstrap=50)
    assert r["casos_com_criterio"] == 0 and "não ocorreu" in frase_resultado(r, "x")


@pytest.mark.parametrize("ic,texto", [((0.02, 0.10), "aumentou a chance"), ((-0.10, -0.02), "DIMINUIU"),
                                      ((-0.05, 0.05), "pode ser acaso")])
def test_leitura_do_intervalo(ic, texto):
    r = {"fiis": 3, "casos": 100, "casos_com_criterio": 20, "taxa_base": 0.5, "taxa_com_criterio": 0.6,
         "periodo": (pd.Timestamp("2019-01-31"), pd.Timestamp("2024-09-30")), "ic90_diferenca": ic}
    assert texto in frase_resultado(r, "critério")


def test_lacuna_de_rendimentos_na_fonte_nao_dispara_o_criterio():
    # rendimentos constantes, mas o provedor "perdeu" 2019-2020; preço estável
    p, _ = _fii(lambda i: 100.0)
    d = pd.Series([1.0 for m in MESES if m.year not in (2019, 2020)],
                  index=[m for m in MESES if m.year not in (2019, 2020)])
    df = avaliar_fii_mes_a_mes(p, d, None)
    # sem a correção, a média de 24 meses ficava baixa após a lacuna e o critério disparava
    assert not df["criterio_yield"].eq(1).any()
    # depois da lacuna, a média só volta a existir ~36 meses após o fim dela (12 do yield + 24 da média)
    depois = df[(df["data"] >= "2021-01-01") & df["criterio_yield"].notna()]
    assert depois["data"].min() >= pd.Timestamp("2023-10-01")
    assert df.loc[df["data"].dt.year == 2020, "yield"].isna().all()


def test_lacuna_nos_12_meses_seguintes_exclui_o_caso():
    p, _ = _fii(lambda i: 100.0)
    d = pd.Series([1.0 for m in MESES if m.year != 2024], index=[m for m in MESES if m.year != 2024])
    df = avaliar_fii_mes_a_mes(p, d, None)
    afetados = df[df["lacuna_futuro"]]
    assert len(afetados) and afetados["superou_cdi"].isna().all()


# ---------------------------------------------------------------- caminho 4 (P/VP, valor patrimonial da CVM)

def test_pvp_com_um_mes_de_defasagem_e_criterios_4():
    # preço cai de 100 para 80 em 2022; VP constante em 100; rendimento constante
    p, d = _fii(lambda i: 100.0 if DATAS[i] < pd.Timestamp("2022-01-01") else 80.0)
    vp = pd.Series(100.0, index=MESES)
    df = avaliar_fii_mes_a_mes(p, d, None, valor_patrimonial=vp)
    linha = df[df["data"] == df["data"][df["data"] >= "2022-03-01"].min()].iloc[0]
    assert linha["pvp"] == pytest.approx(0.8)
    assert linha["criterio_pvp"] == 1.0            # 0,80 < 0,95, rendimentos estáveis
    assert linha["criterio_pvp_historico"] == 1.0  # 0,80 ≤ 90% da média de 24 meses (1,00)
    antes = df[df["data"] < "2021-12-01"]
    assert (antes["criterio_pvp"].dropna() == 0).all()
    # o VP de um mês só é usado a partir do mês seguinte (prazo de publicação do informe)
    vp2 = pd.Series(100.0, index=MESES)
    vp2[vp2.index >= "2023-06-01"] = 50.0
    df2 = avaliar_fii_mes_a_mes(p, d, None, valor_patrimonial=vp2)
    junho = df2[(df2["data"].dt.year == 2023) & (df2["data"].dt.month == 6)].iloc[0]
    julho = df2[(df2["data"].dt.year == 2023) & (df2["data"].dt.month == 7)].iloc[0]
    assert junho["pvp"] == pytest.approx(0.8) and julho["pvp"] == pytest.approx(1.6)


def test_sem_valor_patrimonial_os_criterios_4_ficam_vazios():
    p, d = _fii(lambda i: 100.0)
    df = avaliar_fii_mes_a_mes(p, d, None)
    assert df["pvp"].isna().all() and df["criterio_pvp"].isna().all()
    assert df["criterio_yield_pvp"].isna().all()


# ---------------------------------------------------------------- Etapa C (renda descontada a inflação)

def _fii_ipca(spread_fn, ipca_fn, preco_fn):
    """FII de CRI atrelado ao IPCA: rendimento do mês = VP × (juro real / 12 +
    IPCA do mês de 2 meses antes, a correção monetária do saldo dos CRIs),
    VP constante em 100."""
    ipca = pd.Series([ipca_fn(m) for m in MESES], index=MESES)
    divs = [100 * (spread_fn(m) / 12 + ipca_fn(m - pd.DateOffset(months=2))) for m in MESES]
    d = pd.Series(divs, index=MESES)
    p = pd.Series([preco_fn(dt) for dt in DATAS], index=DATAS, dtype=float)
    return p, d, ipca, pd.Series(100.0, index=MESES)


def test_inflacao_desacelerando_nao_e_queda_de_renda_no_fii_de_papel():
    p, d, ipca, vp = _fii_ipca(lambda m: 0.06, lambda m: 0.008 if m.year < 2023 else 0.0035,
                               lambda dt: 100.0 if dt.year < 2023 else 85.0)
    df = avaliar_fii_mes_a_mes(p, d, None, valor_patrimonial=vp, ipca_mensal=ipca, papel=True)
    fim_2023 = df[(df["data"].dt.year == 2023) & (df["data"].dt.month >= 9)]
    assert len(fim_2023) == 4
    # o rendimento nominal caiu mais de 10% (só porque o IPCA desacelerou) ...
    assert not fim_2023["estavel"].any()
    assert fim_2023["criterio_pvp"].eq(0).all()
    # ... mas a renda acima da inflação ficou estável: as variantes "_real" aceitam
    assert fim_2023["estavel_real"].eq(1).all()
    assert fim_2023["criterio_pvp_real"].eq(1).all()  # P/VP 0,85 com renda real estável
    assert fim_2023["renda_real_12m"].iloc[-1] == pytest.approx(0.06, abs=0.02)


def test_renda_real_em_queda_continua_bloqueando():
    # juro real dos CRIs cai de 6% para 3% (deterioração de verdade), IPCA constante
    p, d, ipca, vp = _fii_ipca(lambda m: 0.06 if m.year < 2023 else 0.03, lambda m: 0.004,
                               lambda dt: 100.0 if dt.year < 2023 else 85.0)
    df = avaliar_fii_mes_a_mes(p, d, None, valor_patrimonial=vp, ipca_mensal=ipca, papel=True)
    fim_2023 = df[(df["data"].dt.year == 2023) & (df["data"].dt.month >= 3)]
    assert fim_2023["estavel_real"].eq(0).all()
    assert not fim_2023["criterio_pvp_real"].eq(1).any()


def test_variantes_reais_iguais_as_nominais_fora_dos_fiis_de_papel():
    p, d, ipca, vp = _fii_ipca(lambda m: 0.06, lambda m: 0.008 if m.year < 2023 else 0.0035,
                               lambda dt: 100.0 if dt.year < 2023 else 85.0)
    df = avaliar_fii_mes_a_mes(p, d, None, valor_patrimonial=vp, ipca_mensal=ipca, papel=False)
    assert df["estavel_real"].isna().all()
    for nominal, real in (("criterio_yield", "criterio_yield_real"), ("criterio_pvp", "criterio_pvp_real"),
                          ("criterio_yield_pvp", "criterio_yield_pvp_real")):
        pd.testing.assert_series_equal(df[nominal], df[real], check_names=False)
    # sem IPCA, o FII de papel também cai na regra nominal
    df2 = avaliar_fii_mes_a_mes(p, d, None, valor_patrimonial=vp, papel=True)
    assert df2["estavel_real"].isna().all()
    pd.testing.assert_series_equal(df2["criterio_pvp"], df2["criterio_pvp_real"], check_names=False)


def test_queda_maxima_ajustavel_para_a_checagem_de_robustez():
    # rendimento cai 8% em 2021: estável com tolerância de 10%, não com 5%
    p = pd.Series(100.0, index=DATAS)
    d = pd.Series([1.0 if m.year < 2021 else 0.92 for m in MESES], index=MESES)
    em_2021 = lambda df: df[df["data"].dt.year == 2021]
    assert em_2021(avaliar_fii_mes_a_mes(p, d, None))["estavel"].all()
    assert not em_2021(avaliar_fii_mes_a_mes(p, d, None, queda_maxima=0.05))["estavel"].all()
    assert em_2021(avaliar_fii_mes_a_mes(p, d, None, queda_maxima=0.15))["estavel"].all()
