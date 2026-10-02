import pandas as pd

import carteira_analise.fontes.bcb as bcb


def test_cdi_em_janelas_de_5_anos(monkeypatch):
    urls = []

    def falso(url, timeout=30):
        urls.append(url)
        return [{"data": "02/01/2020", "valor": "0.017"}, {"data": "03/01/2020", "valor": "0.018"}]

    monkeypatch.setattr(bcb, "_baixar_json", falso)
    s = bcb.buscar_cdi_diario("2014-01-01", "2026-09-30")
    assert len(urls) == 3  # 12,7 anos -> 3 janelas
    assert "dataInicial=01/01/2014" in urls[0]
    assert len(s) == 2 and s.iloc[0] == 0.017  # datas repetidas entre janelas removidas


def test_cdi_falha_devolve_none(monkeypatch):
    def falha(url, timeout=30):
        raise OSError("sem rede")

    monkeypatch.setattr(bcb, "_baixar_json", falha)
    assert bcb.buscar_cdi_diario("2024-01-01", "2024-06-30") is None
    monkeypatch.setattr(bcb, "_baixar_json", lambda url, timeout=30: [])
    assert bcb.buscar_cdi_diario("2024-01-01", "2024-06-30") is None
