import pandas as pd

import carteira_analise.fontes.bcb as bcb


def test_cdi_em_janelas_de_2_anos(monkeypatch):
    urls = []

    def falso(url, timeout=30):
        urls.append(url)
        return [{"data": "02/01/2020", "valor": "0.017"}, {"data": "03/01/2020", "valor": "0.018"}]

    monkeypatch.setattr(bcb, "_baixar_json", falso)
    s = bcb.buscar_cdi_diario("2014-01-01", "2026-09-30")
    assert len(urls) == 7  # 12,7 anos -> 7 janelas de 2 anos
    assert "dataInicial=01/01/2014" in urls[0]
    assert len(s) == 2 and s.iloc[0] == 0.017  # datas repetidas entre janelas removidas


def test_cdi_falha_devolve_none(monkeypatch):
    def falha(url, timeout=30):
        raise OSError("sem rede")

    monkeypatch.setattr(bcb, "_baixar_json", falha)
    esperas = []
    assert bcb.buscar_cdi_diario("2024-01-01", "2024-06-30", dormir=esperas.append) is None
    assert esperas == [2.0, 4.0]  # 3 tentativas, com espera crescente
    monkeypatch.setattr(bcb, "_baixar_json", lambda url, timeout=30: [])
    assert bcb.buscar_cdi_diario("2024-01-01", "2024-06-30") is None


def test_cdi_falha_temporaria_e_recuperada(monkeypatch):
    chamadas = {"n": 0}

    def instavel(url, timeout=45):
        chamadas["n"] += 1
        if chamadas["n"] < 3:
            raise OSError("tempo esgotado")
        return [{"data": "02/01/2024", "valor": "0.043"}]

    monkeypatch.setattr(bcb, "_baixar_json", instavel)
    s = bcb.buscar_cdi_diario("2024-01-01", "2024-03-31", dormir=lambda s: None)
    assert s is not None and s.iloc[0] == 0.043 and chamadas["n"] == 3
