from __future__ import annotations

import pytest

from carteira_analise.fontes.fundamentus import (
    FundamentosFII,
    _parse_numero_br,
    buscar_fundamentos_fii,
    buscar_todos_fiis_fundamentus,
)


HTML_AMOSTRA = """
<table id="tabelaResultado">
<thead><tr>
<th>Papel</th><th>Segmento</th><th>Cotação</th><th>FFO Yield</th>
<th>Dividend Yield</th><th>P/VP</th><th>Valor de Mercado</th><th>Liquidez</th>
<th>Qtd de imóveis</th><th>Preço do m2</th><th>Aluguel por m2</th>
<th>Cap Rate</th><th>Vacância Média</th><th>Endereço</th>
</tr></thead>
<tbody>
<tr><td>BTLG11</td><td>Multicategoria</td><td>98,91</td><td>6,09%</td><td>8,63%</td><td>0,97</td>
<td>6.863.330.000</td><td>14.368.600</td><td>31</td><td>3.360,87</td><td>258,46</td>
<td>7,69%</td><td>5,06%</td><td>endereco...</td></tr>
<tr><td>HGLG11</td><td>Logística</td><td>158,20</td><td>8,50%</td><td>9,10%</td><td>1,02</td>
<td>5.200.000.000</td><td>8.500.000</td><td>18</td><td>2.900,00</td><td>210,00</td>
<td>7,90%</td><td>3,20%</td><td>endereco...</td></tr>
<tr><td>KNIP11</td><td>Títulos e Val. Mob.</td><td>92,50</td><td>0,00%</td><td>13,20%</td><td>0,88</td>
<td>1.200.000.000</td><td>500.000</td><td>0</td><td>0,00</td><td>0,00</td>
<td>0,00%</td><td>0,00%</td><td></td></tr>
</tbody>
</table>
"""


class TestParseNumeroBr:
    def test_decimal_simples(self):
        assert _parse_numero_br("43,49") == pytest.approx(43.49)

    def test_percentual_vira_fracao(self):
        assert _parse_numero_br("10,80%") == pytest.approx(0.108)

    def test_milhar_com_ponto(self):
        assert _parse_numero_br("400.913.000") == pytest.approx(400913000.0)

    def test_zero_percentual(self):
        assert _parse_numero_br("0,00%") == pytest.approx(0.0)

    def test_traco_vira_none(self):
        assert _parse_numero_br("-") is None

    def test_vazio_vira_none(self):
        assert _parse_numero_br("") is None

    def test_none_vira_none(self):
        assert _parse_numero_br(None) is None

    def test_float_ja_convertido_pelo_pandas_nao_e_reprocessado(self):
        # Regressão: pd.read_html às vezes já converte a coluna para float
        # (quando não tem '%' misturado) — nesse caso, NÃO deve reprocessar
        # como se fosse texto (bug real encontrado: 98.91 virava 9891.0)
        assert _parse_numero_br(98.91) == pytest.approx(98.91)
        assert _parse_numero_br(0.97) == pytest.approx(0.97)

    def test_int_e_aceito(self):
        assert _parse_numero_br(31) == pytest.approx(31.0)

    def test_nan_float_vira_none(self):
        import math
        assert _parse_numero_br(math.nan) is None


class TestBuscarTodosFiisFundamentus:
    def test_parseia_tabela_completa(self, monkeypatch):
        import carteira_analise.fontes.fundamentus as mod
        monkeypatch.setattr(mod, "_baixar_html_fii_resultado", lambda: HTML_AMOSTRA)

        resultado = buscar_todos_fiis_fundamentus()

        assert set(resultado.keys()) == {"BTLG11", "HGLG11", "KNIP11"}
        assert isinstance(resultado["BTLG11"], FundamentosFII)

    def test_valores_numericos_corretos(self, monkeypatch):
        import carteira_analise.fontes.fundamentus as mod
        monkeypatch.setattr(mod, "_baixar_html_fii_resultado", lambda: HTML_AMOSTRA)

        resultado = buscar_todos_fiis_fundamentus()
        btlg = resultado["BTLG11"]

        assert btlg.segmento == "Multicategoria"
        assert btlg.cotacao == pytest.approx(98.91)
        assert btlg.dividend_yield == pytest.approx(0.0863)
        assert btlg.pvp == pytest.approx(0.97)
        assert btlg.qtd_imoveis == 31
        assert btlg.cap_rate == pytest.approx(0.0769)
        assert btlg.vacancia_media == pytest.approx(0.0506)

    def test_fii_de_papel_com_zero_imoveis(self, monkeypatch):
        # KNIP11 é um FII de papel (títulos), não de tijolo — não tem
        # imóveis físicos, o que é um caso legítimo, não um erro
        import carteira_analise.fontes.fundamentus as mod
        monkeypatch.setattr(mod, "_baixar_html_fii_resultado", lambda: HTML_AMOSTRA)

        resultado = buscar_todos_fiis_fundamentus()
        knip = resultado["KNIP11"]
        assert knip.qtd_imoveis == 0
        assert knip.dividend_yield == pytest.approx(0.132)


class TestBuscarFundamentosFii:
    def test_busca_ticker_especifico_com_sufixo_sa(self, monkeypatch):
        import carteira_analise.fontes.fundamentus as mod
        monkeypatch.setattr(mod, "_baixar_html_fii_resultado", lambda: HTML_AMOSTRA)
        monkeypatch.setattr(mod, "_cache_tabela_completa", None)

        resultado = buscar_fundamentos_fii("BTLG11.SA")
        assert resultado is not None
        assert resultado.ticker == "BTLG11"

    def test_ticker_inexistente_devolve_none(self, monkeypatch):
        import carteira_analise.fontes.fundamentus as mod
        monkeypatch.setattr(mod, "_baixar_html_fii_resultado", lambda: HTML_AMOSTRA)
        monkeypatch.setattr(mod, "_cache_tabela_completa", None)

        resultado = buscar_fundamentos_fii("FANTASMA11.SA")
        assert resultado is None

    def test_falha_de_rede_devolve_none_sem_quebrar(self, monkeypatch):
        import carteira_analise.fontes.fundamentus as mod

        def _falha():
            raise ConnectionError("rede indisponível")

        monkeypatch.setattr(mod, "_baixar_html_fii_resultado", _falha)
        monkeypatch.setattr(mod, "_cache_tabela_completa", None)

        resultado = buscar_fundamentos_fii("BTLG11.SA")
        assert resultado is None

    def test_cache_evita_segunda_chamada_de_rede(self, monkeypatch):
        import carteira_analise.fontes.fundamentus as mod
        chamadas = {"n": 0}

        def _contar():
            chamadas["n"] += 1
            return HTML_AMOSTRA

        monkeypatch.setattr(mod, "_baixar_html_fii_resultado", _contar)
        monkeypatch.setattr(mod, "_cache_tabela_completa", None)

        buscar_fundamentos_fii("BTLG11.SA")
        buscar_fundamentos_fii("HGLG11.SA")
        assert chamadas["n"] == 1
