from __future__ import annotations

from carteira_analise.carteira import parse_tickers


class TestParseTickers:
    def test_lista_simples_bem_formatada(self):
        assert parse_tickers("PETR4.SA, ITUB4.SA, WEGE3.SA") == [
            "PETR4.SA",
            "ITUB4.SA",
            "WEGE3.SA",
        ]

    def test_ignora_virgula_final_solta(self):
        assert parse_tickers("PETR4.SA, ITUB4.SA,") == ["PETR4.SA", "ITUB4.SA"]

    def test_ignora_espacos_extras(self):
        assert parse_tickers("  PETR4.SA ,  ITUB4.SA  ") == ["PETR4.SA", "ITUB4.SA"]

    def test_corrige_virgula_no_lugar_do_ponto_antes_do_sufixo(self):
        # Regressão do bug real: digitar "VALE3,SA" ao invés de "VALE3.SA"
        # gerava uma linha fantasma "SA" na tabela de resultados.
        assert parse_tickers("VALE3,SA, ABCB4.SA") == ["VALE3.SA", "ABCB4.SA"]

    def test_corrige_multiplas_ocorrencias_do_erro(self):
        resultado = parse_tickers("VALE3,SA, ABCB4.SA, BBSE3.SA, TRXF11,SA, JURO11.SA")
        assert resultado == [
            "VALE3.SA",
            "ABCB4.SA",
            "BBSE3.SA",
            "TRXF11.SA",
            "JURO11.SA",
        ]

    def test_sufixo_us_tambem_e_corrigido(self):
        assert parse_tickers("AAPL,US") == ["AAPL.US"]

    def test_lista_vazia(self):
        assert parse_tickers("") == []

    def test_converte_para_maiusculas(self):
        assert parse_tickers("petr4.sa") == ["PETR4.SA"]
