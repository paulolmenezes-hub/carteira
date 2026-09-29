from __future__ import annotations

import time

import pandas as pd

from carteira_analise.fontes.cache import FonteComCache


class _FonteFalsa:
    """Fonte de dados falsa que conta quantas vezes cada método foi chamado —
    usada pra provar que o cache evita chamadas repetidas, sem precisar de
    rede nem de um servidor de teste."""

    def __init__(self) -> None:
        self.chamadas_precos = 0
        self.chamadas_info = 0
        self.chamadas_dividendos = 0

    def baixar_precos(self, ticker: str, periodo: str) -> pd.Series:
        self.chamadas_precos += 1
        datas = pd.bdate_range("2024-08-01", periods=10)
        return pd.Series([10.5 + i * 0.37 for i in range(10)], index=datas, dtype=float)

    def baixar_info(self, ticker: str) -> dict:
        self.chamadas_info += 1
        return {"trailingPE": 10.0, "priceToBook": 1.5}

    def baixar_dividendos(self, ticker: str) -> pd.Series:
        self.chamadas_dividendos += 1
        datas = pd.bdate_range("2024-08-01", periods=3)
        return pd.Series([0.5, 0.5, 0.5], index=datas)


class _FonteFalsaVazia:
    """Fonte que sempre retorna vazio — simula ticker inexistente ou falha
    transitória de rede, pra testar que resultados vazios não são cacheados."""

    def __init__(self) -> None:
        self.chamadas = 0

    def baixar_precos(self, ticker: str, periodo: str) -> pd.Series:
        self.chamadas += 1
        return pd.Series(dtype=float)

    def baixar_info(self, ticker: str) -> dict:
        self.chamadas += 1
        return {}

    def baixar_dividendos(self, ticker: str) -> pd.Series:
        self.chamadas += 1
        return pd.Series(dtype=float)


class TestFonteComCachePrecos:
    def test_segunda_chamada_nao_bate_na_fonte_original(self, tmp_path):
        fonte_falsa = _FonteFalsa()
        cache = FonteComCache(fonte_falsa, caminho_db=tmp_path / "teste.sqlite")

        cache.baixar_precos("PETR4.SA", "2y")
        cache.baixar_precos("PETR4.SA", "2y")

        assert fonte_falsa.chamadas_precos == 1

    def test_tickers_diferentes_nao_compartilham_cache(self, tmp_path):
        fonte_falsa = _FonteFalsa()
        cache = FonteComCache(fonte_falsa, caminho_db=tmp_path / "teste.sqlite")

        cache.baixar_precos("PETR4.SA", "2y")
        cache.baixar_precos("VALE3.SA", "2y")

        assert fonte_falsa.chamadas_precos == 2

    def test_periodos_diferentes_nao_compartilham_cache(self, tmp_path):
        fonte_falsa = _FonteFalsa()
        cache = FonteComCache(fonte_falsa, caminho_db=tmp_path / "teste.sqlite")

        cache.baixar_precos("PETR4.SA", "1y")
        cache.baixar_precos("PETR4.SA", "2y")

        assert fonte_falsa.chamadas_precos == 2

    def test_dado_cacheado_bate_com_o_original(self, tmp_path):
        fonte_falsa = _FonteFalsa()
        cache = FonteComCache(fonte_falsa, caminho_db=tmp_path / "teste.sqlite")

        original = fonte_falsa.baixar_precos("PETR4.SA", "2y")
        do_cache = cache.baixar_precos("PETR4.SA", "2y")  # 1ª chamada no cache, vai na fonte
        do_cache_de_novo = cache.baixar_precos("PETR4.SA", "2y")  # 2ª, deve vir do cache

        pd.testing.assert_series_equal(do_cache, do_cache_de_novo, check_names=False, check_freq=False)
        assert list(do_cache.values) == list(original.values)

    def test_ttl_expirado_busca_de_novo(self, tmp_path):
        fonte_falsa = _FonteFalsa()
        cache = FonteComCache(
            fonte_falsa, caminho_db=tmp_path / "teste.sqlite", ttl_horas=0
        )  # TTL zero = sempre expirado

        cache.baixar_precos("PETR4.SA", "2y")
        time.sleep(0.01)
        cache.baixar_precos("PETR4.SA", "2y")

        assert fonte_falsa.chamadas_precos == 2

    def test_resultado_vazio_nao_e_cacheado(self, tmp_path):
        fonte_vazia = _FonteFalsaVazia()
        cache = FonteComCache(fonte_vazia, caminho_db=tmp_path / "teste.sqlite")

        cache.baixar_precos("TICKER_INEXISTENTE.SA", "2y")
        cache.baixar_precos("TICKER_INEXISTENTE.SA", "2y")

        # sem cache de falha: bateu na fonte as duas vezes
        assert fonte_vazia.chamadas == 2


class TestFonteComCacheInfoEDividendos:
    def test_info_e_cacheado(self, tmp_path):
        fonte_falsa = _FonteFalsa()
        cache = FonteComCache(fonte_falsa, caminho_db=tmp_path / "teste.sqlite")

        cache.baixar_info("PETR4.SA")
        cache.baixar_info("PETR4.SA")

        assert fonte_falsa.chamadas_info == 1

    def test_dividendos_e_cacheado(self, tmp_path):
        fonte_falsa = _FonteFalsa()
        cache = FonteComCache(fonte_falsa, caminho_db=tmp_path / "teste.sqlite")

        cache.baixar_dividendos("PETR4.SA")
        cache.baixar_dividendos("PETR4.SA")

        assert fonte_falsa.chamadas_dividendos == 1

    def test_info_vazio_nao_e_cacheado(self, tmp_path):
        fonte_vazia = _FonteFalsaVazia()
        cache = FonteComCache(fonte_vazia, caminho_db=tmp_path / "teste.sqlite")

        cache.baixar_info("TICKER_INEXISTENTE.SA")
        cache.baixar_info("TICKER_INEXISTENTE.SA")

        assert fonte_vazia.chamadas == 2


class TestLimparCache:
    def test_limpar_remove_tudo(self, tmp_path):
        fonte_falsa = _FonteFalsa()
        cache = FonteComCache(fonte_falsa, caminho_db=tmp_path / "teste.sqlite")

        cache.baixar_precos("PETR4.SA", "2y")
        cache.limpar()
        cache.baixar_precos("PETR4.SA", "2y")

        assert fonte_falsa.chamadas_precos == 2


class TestGerenciadorDeContexto:
    def test_with_fecha_conexao_automaticamente(self, tmp_path):
        fonte_falsa = _FonteFalsa()
        with FonteComCache(fonte_falsa, caminho_db=tmp_path / "teste.sqlite") as cache:
            cache.baixar_precos("PETR4.SA", "2y")
        # conexão fechada; nova tentativa de uso deve falhar
        try:
            cache._conn.execute("SELECT 1")
            levantou_erro = False
        except Exception:
            levantou_erro = True
        assert levantou_erro
