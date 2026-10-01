"""Camada de rede do Yahoo Finance, com o yfinance simulado."""
import pandas as pd

import carteira_analise.fontes.yahoo as yahoo


def test_baixar_precos_sem_ajuste_por_proventos(monkeypatch):
    chamadas = {}

    def download_falso(ticker, **kwargs):
        chamadas.update(kwargs)
        return pd.DataFrame({"Close": [10.0, None, 11.0]}, index=pd.bdate_range("2024-01-01", periods=3))

    monkeypatch.setattr(yahoo.yf, "download", download_falso)
    s = yahoo.baixar_precos("HGLG11.SA", "5y")
    assert chamadas["auto_adjust"] is False  # proventos não podem vir embutidos no preço
    assert list(s.values) == [10.0, 11.0]


def test_baixar_precos_vazio_e_multicoluna(monkeypatch):
    monkeypatch.setattr(yahoo.yf, "download", lambda t, **k: pd.DataFrame())
    assert yahoo.baixar_precos("X", "1y").empty
    idx = pd.bdate_range("2024-01-01", periods=2)
    multi = pd.DataFrame({("Close", "X"): [1.0, 2.0]}, index=idx)
    monkeypatch.setattr(yahoo.yf, "download", lambda t, **k: multi)
    assert list(yahoo.baixar_precos("X", "1y").values) == [1.0, 2.0]


def test_info_e_dividendos_com_falha(monkeypatch):
    class TickerFalha:
        def __init__(self, t):
            pass

        @property
        def info(self):
            raise RuntimeError("rede")

        @property
        def dividends(self):
            raise RuntimeError("rede")

    monkeypatch.setattr(yahoo.yf, "Ticker", TickerFalha)
    assert yahoo.baixar_info("X") == {}
    assert yahoo.baixar_dividendos("X").empty


def test_info_e_dividendos_ok(monkeypatch):
    class TickerOk:
        def __init__(self, t):
            self.info = {"trailingPE": 10.0}
            self.dividends = pd.Series([1.0], index=pd.bdate_range("2024-01-01", periods=1))

    monkeypatch.setattr(yahoo.yf, "Ticker", TickerOk)
    assert yahoo.baixar_info("X") == {"trailingPE": 10.0}
    assert len(yahoo.baixar_dividendos("X")) == 1
