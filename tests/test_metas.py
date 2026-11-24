import pytest

from carteira_analise.metas import CLASSES, comparar, dividir_aporte, metas_de_texto, metas_para_texto

VALORES = {"Ações B3": 176402, "FIIs": 107998, "ETFs B3": 21250, "EUA (ações e ETFs)": 59096}
METAS = {"Ações B3": 45, "FIIs": 30, "ETFs B3": 10, "EUA (ações e ETFs)": 15}


def test_comparar():
    l = {x["classe"]: x for x in comparar(VALORES, METAS)}
    total = sum(VALORES.values())
    assert l["Ações B3"]["hoje"] == pytest.approx(176402 / total * 100)
    assert l["ETFs B3"]["falta"] == pytest.approx(total * 0.10 - 21250)
    assert not l["ETFs B3"]["na_meta"] and l["FIIs"]["na_meta"]
    assert sum(x["falta"] for x in l.values()) == pytest.approx(0, abs=1e-6)


def test_dividir_aporte_so_nas_classes_abaixo():
    linhas = comparar(VALORES, METAS)
    d = dividir_aporte(linhas, 10000)
    assert sum(d.values()) == pytest.approx(10000)
    assert d["Ações B3"] == 0 and d["ETFs B3"] > d["FIIs"] >= 0


def test_aporte_maior_que_a_falta_distribui_o_resto_pelas_metas():
    linhas = comparar({"Ações B3": 50, "FIIs": 50, "ETFs B3": 0, "EUA (ações e ETFs)": 0},
                      {"Ações B3": 50, "FIIs": 50, "ETFs B3": 0, "EUA (ações e ETFs)": 0})
    assert dividir_aporte(linhas, 100) == pytest.approx({"Ações B3": 50, "FIIs": 50, "ETFs B3": 0,
                                                          "EUA (ações e ETFs)": 0})


def test_texto_ida_e_volta():
    assert metas_de_texto(metas_para_texto(METAS)) == {c: float(METAS[c]) for c in CLASSES}
    assert metas_de_texto("50-30-10") is None and metas_de_texto("50-30-10-20") is None
    assert metas_de_texto(None) is None


def test_limite_por_ativo():
    from carteira_analise.metas import (LIMITES_PADRAO, limites_de_texto, limites_para_texto, pesos_dos_ativos,
                                        situacao_peso)
    assert limites_de_texto("10-0") == {"maximo": 10.0, "minimo": 0.0}
    assert limites_de_texto("6-2.5") == {"maximo": 6.0, "minimo": 2.5}
    assert limites_de_texto("5-8") is None and limites_de_texto("abc") is None and limites_de_texto(None) is None
    assert limites_para_texto({"maximo": 6.0, "minimo": 2.5}) == "6-2.5"
    s = situacao_peso(7200, 100000, {"maximo": 6, "minimo": 2})
    assert s["situacao"] == "acima" and abs(s["diferenca"] - 1200) < 1e-9
    s = situacao_peso(1100, 100000, {"maximo": 6, "minimo": 2})
    assert s["situacao"] == "abaixo" and abs(s["diferenca"] - 900) < 1e-9
    assert situacao_peso(1100, 100000, LIMITES_PADRAO)["situacao"] == "dentro"  # mínimo 0 = desligado
    p = pesos_dos_ativos({"A": 30.0, "B": 70.0, "C": 0.0}, {"maximo": 50, "minimo": 0})
    assert [l["ticker"] for l in p] == ["B", "A"] and p[0]["situacao"] == "acima"
