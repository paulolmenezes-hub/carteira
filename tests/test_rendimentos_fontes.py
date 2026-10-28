"""Conferência cruzada dos rendimentos de FIIs entre fontes."""
import pandas as pd
import pytest

from carteira_analise.fontes.rendimentos import (
    comparar_mes_a_mes,
    conferir_fontes,
    rendimentos_por_mes,
    resumo_conferencia,
)


def _mensal(valores, inicio="2022-01-10"):
    return pd.Series(valores, index=pd.date_range(inicio, periods=len(valores), freq="MS") + pd.Timedelta(days=9))


def test_rendimentos_por_mes_soma_e_ignora_zeros_e_fuso():
    s = pd.Series([0.5, 0.3, 0.0, float("nan")],
                  index=pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-05", "2024-03-05"]).tz_localize("UTC"))
    m = rendimentos_por_mes(s)
    assert list(m.index.astype(str)) == ["2024-01"]
    assert m.iloc[0] == pytest.approx(0.8)
    assert rendimentos_por_mes(None).empty


def test_serie_constante_fica_sem_defasagem():
    assert comparar_mes_a_mes(_mensal([1.0] * 24), _mensal([1.0] * 24, inicio="2022-02-10"))["defasagem"] == 0


def test_comparar_concorda_com_arredondamento():
    a = _mensal([1.00] * 24)
    b = _mensal([1.02] * 24)  # 2% de diferença: dentro da tolerância
    c = comparar_mes_a_mes(a, b)
    assert c["meses_comuns"] == 24 and c["taxa"] == 1.0 and c["defasagem"] == 0


def test_comparar_acha_defasagem_de_um_mes():
    valores = [1.0 + 0.1 * (i % 5) for i in range(24)]
    a = _mensal(valores)
    b = _mensal(valores, inicio="2022-02-10")  # mesma série, um mês depois (mês de referência x data-com)
    c = comparar_mes_a_mes(a, b)
    assert c["defasagem"] == -1 and c["taxa"] == 1.0


def test_lacunas_nao_contam_como_divergencia():
    a = _mensal([1.0] * 24)
    b = a.iloc[::2]  # Yahoo com metade dos meses
    c = comparar_mes_a_mes(a, b)
    assert c["meses_comuns"] == 12 and c["taxa"] == 1.0


def test_conferir_confirmado_pela_segunda_fonte():
    r = conferir_fontes({"fundamentus": _mensal([1.0] * 24), "cvm": _mensal([1.01] * 24), "yahoo": None})
    assert r["usada"] == "fundamentus" and r["selo"] == "confirmado"
    assert r["confirmada_por"] == ["cvm"]
    assert r["disponiveis"] == ["fundamentus", "cvm"]
    assert "conferido com CVM" in r["texto"]


def test_conferir_cai_para_proxima_fonte_quando_a_primeira_falha():
    r = conferir_fontes({"fundamentus": pd.Series(dtype=float), "cvm": _mensal([1.0] * 24),
                         "yahoo": _mensal([1.0] * 24)})
    assert r["usada"] == "cvm" and r["confirmada_por"] == ["yahoo"]


def test_conferir_divergente():
    r = conferir_fontes({"fundamentus": _mensal([1.0] * 24), "cvm": _mensal([1.5] * 24)})
    assert r["selo"] == "divergente" and "diverge de CVM" in r["texto"]


def test_conferir_fonte_unica_quando_outra_tem_poucos_meses():
    r = conferir_fontes({"fundamentus": _mensal([1.0] * 24), "yahoo": _mensal([1.0] * 5)})
    assert r["selo"] == "fonte_unica"


def test_conferir_sem_dados():
    r = conferir_fontes({"fundamentus": None, "cvm": pd.Series(dtype=float)})
    assert r["selo"] == "sem_dados" and r["usada"] is None


def test_resumo_conferencia_colunas_fixas():
    valores = [1.0 + 0.1 * (i % 5) for i in range(24)]
    r = conferir_fontes({"fundamentus": _mensal(valores), "cvm": _mensal(valores, inicio="2022-02-10")})
    linha = resumo_conferencia(r)
    assert linha["Fonte usada"] == "Fundamentus"
    assert linha["Conferência"] == "✅ confirmado"
    assert linha["Concorda c/ Fundamentus"] == "(usada)"
    assert linha["Concorda c/ CVM"] == "100% de 12 janelas de 12m"
    assert linha["Concorda c/ Yahoo"] == "—"


def test_cvm_comparada_pela_soma_de_12_meses():
    fund = _mensal([1.0] * 36)
    # CVM com o mesmo total anual, mas distribuído em meses diferentes (competência x pagamento)
    cvm = _mensal([0.7, 1.3] * 18)
    assert comparar_mes_a_mes(fund, cvm)["taxa"] < 0.5
    r = conferir_fontes({"fundamentus": fund, "cvm": cvm})
    assert r["comparacoes"]["cvm"]["metodo"] == "soma12"
    assert r["selo"] == "confirmado"
    assert "janelas de 12m" in resumo_conferencia(r)["Concorda c/ CVM"]


def test_somas_12m_ignoram_janelas_com_lacuna():
    from carteira_analise.fontes.rendimentos import comparar_somas_12m
    fund = _mensal([1.0] * 36)
    yahoo = fund.drop(fund.index[[5, 7, 20]])  # meses faltando: janelas com 2+ lacunas não entram
    r = comparar_somas_12m(fund, yahoo)
    assert r["taxa"] == 1.0


def test_diagnostico_de_desdobramento_nao_ajustado():
    from carteira_analise.fontes.rendimentos import diagnosticar_escala
    yahoo = _mensal([0.1] * 48)
    fund = yahoo.copy()
    fund.iloc[:24] = 1.0  # Fundamentus sem ajuste antes do desdobramento de 1 para 10
    d = diagnosticar_escala(fund, yahoo)
    assert d == {"fator": 10.0, "ate": "2024-01"}
    r = conferir_fontes({"fundamentus": fund, "yahoo": yahoo}, ajustar_escala=False)
    assert r["selo"] == "divergente"
    assert "10x maiores" in r["texto"] and "01/2024" in r["texto"]
    assert "desdobramento" in resumo_conferencia(r)["Observação"]


def test_sem_diagnostico_quando_a_divergencia_e_espalhada():
    from carteira_analise.fontes.rendimentos import diagnosticar_escala
    yahoo = _mensal([1.0] * 48)
    fund = _mensal([1.0, 1.3] * 24)
    assert diagnosticar_escala(fund, yahoo) is None


def test_escala_corrigida_pelo_yahoo_e_conferencia_refeita():
    yahoo = _mensal([0.1] * 48)
    fund = yahoo.copy()
    fund.iloc[:24] = 1.0  # Fundamentus e CVM com o valor por cota de antes do desdobramento 1:10
    cvm = fund.copy()
    r = conferir_fontes({"fundamentus": fund, "cvm": cvm, "yahoo": yahoo})
    assert r["ajuste_escala"] == {"fator": 10.0, "ate": "2024-01"}
    assert r["serie"].iloc[0] == pytest.approx(0.1) and r["serie"].iloc[-1] == pytest.approx(0.1)
    assert r["selo"] == "confirmado" and set(r["confirmada_por"]) == {"cvm", "yahoo"}
    assert r["observacoes"][0].startswith("escala corrigida: valores do Fundamentus e da CVM até 01/2024 divididos por 10")
    sem = conferir_fontes({"fundamentus": fund, "yahoo": yahoo}, ajustar_escala=False)
    assert sem["ajuste_escala"] is None and sem["selo"] == "divergente"


def _cpts():
    """CPTS11: desdobramento de 1 para 10 em 09/2023; Fundamentus e CVM com o
    valor por cota da época; Yahoo com poucos rendimentos antes do evento."""
    fund = _mensal([1.0] * 30 + [0.1] * 30, inicio="2021-01-10")
    yahoo = fund[fund.index >= "2023-06-01"] / 1.0
    yahoo[yahoo.index < "2023-08-01"] = 0.1
    eventos = pd.Series([10.0], index=[pd.Timestamp("2023-08-01")])
    return fund, yahoo, eventos


def test_ajustar_por_desdobramentos_divide_antes_do_evento():
    from carteira_analise.fontes.rendimentos import ajustar_por_desdobramentos
    fund, _, eventos = _cpts()
    s, aplicados = ajustar_por_desdobramentos(fund, eventos)
    assert aplicados == [(pd.Timestamp("2023-08-01"), 10.0)]
    assert s.max() == pytest.approx(0.1) and s.min() == pytest.approx(0.1)


def test_ajustar_por_desdobramentos_nao_mexe_em_serie_ja_ajustada_nem_em_bonificacao():
    from carteira_analise.fontes.rendimentos import ajustar_por_desdobramentos
    ajustada = _mensal([0.1] * 60, inicio="2021-01-10")
    s, aplicados = ajustar_por_desdobramentos(ajustada, pd.Series([10.0], index=[pd.Timestamp("2023-08-01")]))
    assert aplicados == [] and s.equals(ajustada.astype(float))
    _, aplicados = ajustar_por_desdobramentos(ajustada, pd.Series([1.1], index=[pd.Timestamp("2023-08-01")]))
    assert aplicados == []


def test_conferir_com_eventos_do_yahoo():
    fund, yahoo, eventos = _cpts()
    cvm = fund.copy()
    r = conferir_fontes({"fundamentus": fund, "cvm": cvm, "yahoo": yahoo}, eventos=eventos)
    assert r["serie"].max() == pytest.approx(0.1)
    assert r["selo"] == "confirmado" and "cvm" in r["confirmada_por"]
    assert r["observacoes"][0].startswith("desdobramento de 1 para 10 em 08/2023 (Yahoo)")
