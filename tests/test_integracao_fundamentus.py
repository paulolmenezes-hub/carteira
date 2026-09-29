from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carteira_analise.carteira import analisar_ativo
from carteira_analise.fontes.fundamentus import FundamentosFII


class _FonteFalsa:
    """Fonte de dados sintética, sem I/O de rede — mesmo padrão usado em
    test_cache.py, aqui reaproveitado para testar a integração do
    Fundamentus dentro de analisar_ativo()."""

    def __init__(self, info=None, dividendos=None):
        self._info = info or {}
        self._dividendos = dividendos if dividendos is not None else pd.Series(dtype=float)

    def baixar_precos(self, ticker, periodo):
        rng = np.random.default_rng(42)
        datas = pd.bdate_range("2023-01-01", periods=300)
        precos = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300))
        return pd.Series(precos, index=datas)

    def baixar_info(self, ticker):
        return self._info

    def baixar_dividendos(self, ticker):
        return self._dividendos


class TestIntegracaoFundamentusEmAnalisarAtivo:
    def test_fii_usa_pvp_do_fundamentus_quando_disponivel(self, monkeypatch):
        import carteira_analise.carteira as mod_carteira

        fundamentos_fake = FundamentosFII(
            ticker="BTLG11", segmento="Multicategoria", cotacao=98.91,
            ffo_yield=0.0609, dividend_yield=0.0863, pvp=0.97,
            valor_mercado=6_863_330_000.0, liquidez=14_368_600.0,
            qtd_imoveis=31, cap_rate=0.0769, vacancia_media=0.0506,
        )
        monkeypatch.setattr(
            "carteira_analise.fontes.fundamentus.buscar_fundamentos_fii",
            lambda ticker: fundamentos_fake,
        )

        fonte = _FonteFalsa(info={"priceToBook": 5.0})  # valor do Yahoo, deveria ser IGNORADO
        resultado = analisar_ativo("BTLG11.SA", "fii", "2y", fonte)

        assert resultado is not None
        # O P/VP usado deve ser o do Fundamentus (0.97), não o do Yahoo (5.0)
        assert resultado.fund["pvp"] == pytest.approx(0.97)
        assert resultado.fund["segmento"] == "Multicategoria"
        assert resultado.fund["vacancia_media"] == pytest.approx(0.0506)

    def test_fii_cai_para_yahoo_quando_fundamentus_falha(self, monkeypatch):
        def _falha(ticker):
            raise ConnectionError("Fundamentus indisponível")

        monkeypatch.setattr(
            "carteira_analise.fontes.fundamentus.buscar_fundamentos_fii", _falha
        )

        fonte = _FonteFalsa(info={"priceToBook": 1.05})
        resultado = analisar_ativo("HGLG11.SA", "fii", "2y", fonte)

        assert resultado is not None
        # Sem Fundamentus disponível, o P/VP do Yahoo deve ser mantido (fallback)
        assert resultado.fund["pvp"] == pytest.approx(1.05)
        assert "segmento" not in resultado.fund

    def test_fii_cai_para_yahoo_quando_ticker_nao_encontrado_no_fundamentus(self, monkeypatch):
        monkeypatch.setattr(
            "carteira_analise.fontes.fundamentus.buscar_fundamentos_fii",
            lambda ticker: None,
        )

        fonte = _FonteFalsa(info={"priceToBook": 0.85})
        resultado = analisar_ativo("FANTASMA11.SA", "fii", "2y", fonte)

        assert resultado is not None
        assert resultado.fund["pvp"] == pytest.approx(0.85)

    def test_acao_nao_aciona_fundamentus(self, monkeypatch):
        # Fundamentus é usado só para FIIs — para ações, nem deveria ser chamado
        chamadas = {"n": 0}

        def _contar(ticker):
            chamadas["n"] += 1
            return None

        monkeypatch.setattr(
            "carteira_analise.fontes.fundamentus.buscar_fundamentos_fii", _contar
        )

        fonte = _FonteFalsa(info={"priceToBook": 2.0, "trailingPE": 12.0})
        analisar_ativo("PETR4.SA", "acao", "2y", fonte)

        assert chamadas["n"] == 0
