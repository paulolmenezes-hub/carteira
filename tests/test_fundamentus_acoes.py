"""Fonte fundamentalista do Fundamentus para ações B3 (P/L, P/VP, ROE,
ROIC, margem líquida, dívida bruta/patrimônio, liquidez corrente) e sua
integração em analisar_ativo(), com fallback para o Yahoo Finance. Sem
rede: o HTML da tabela é uma amostra no mesmo formato do site."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import carteira_analise.fontes.fundamentus as fdm
from carteira_analise.carteira import analisar_ativo
from carteira_analise.fontes.fundamentus import (
    FundamentosAcao,
    _achar_coluna,
    buscar_fundamentos_acao,
    buscar_todas_acoes_fundamentus,
)

HTML_ACOES = """
<table id="resultado">
<thead><tr>
<th>Papel</th><th>Cotação</th><th>P/L</th><th>P/VP</th><th>PSR</th><th>Div.Yield</th>
<th>P/Ativo</th><th>P/Cap.Giro</th><th>P/EBIT</th><th>P/Ativ Circ.Liq</th><th>EV/EBIT</th>
<th>EV/EBITDA</th><th>Mrg Ebit</th><th>Mrg. Líq.</th><th>Liq. Corr.</th><th>ROIC</th><th>ROE</th>
<th>Liq.2meses</th><th>Patrim. Líq</th><th>Dív.Brut/ Patrim.</th><th>Cresc. Rec.5a</th>
</tr></thead>
<tbody>
<tr><td>VALE3</td><td>71,08</td><td>8,50</td><td>1,40</td><td>1,5</td><td>9,80%</td><td>0,6</td><td>5,0</td>
<td>4,0</td><td>-1,0</td><td>4,5</td><td>3,9</td><td>30,00%</td><td>18,50%</td><td>1,35</td><td>15,20%</td>
<td>17,80%</td><td>2.000.000.000</td><td>190.000.000.000</td><td>0,45</td><td>8,00%</td></tr>
<tr><td>BBAS3</td><td>25,10</td><td>4,20</td><td>0,75</td><td>0,5</td><td>10,50%</td><td>0,1</td><td>0,0</td>
<td>0,0</td><td>0,0</td><td>0,0</td><td>0,0</td><td>0,00%</td><td>20,00%</td><td>-</td><td>-</td>
<td>19,90%</td><td>900.000.000</td><td>180.000.000.000</td><td>-</td><td>12,00%</td></tr>
<tr><td> pomo4 </td><td>4,18</td><td>6,00</td><td>1,10</td><td>0,4</td><td>7,00%</td><td>0,5</td><td>2,0</td>
<td>5,0</td><td>-2,0</td><td>5,5</td><td>4,5</td><td>9,00%</td><td>7,50%</td><td>1,80</td><td>11,00%</td>
<td>18,00%</td><td>30.000.000</td><td>3.000.000.000</td><td>0,30</td><td>15,00%</td></tr>
</tbody>
</table>
"""


@pytest.fixture(autouse=True)
def _limpar_cache(monkeypatch):
    monkeypatch.setattr(fdm, "_cache_tabela_acoes", None)


@pytest.fixture
def html_falso(monkeypatch):
    chamadas = []

    def baixar():
        chamadas.append(1)
        return HTML_ACOES

    monkeypatch.setattr(fdm, "_baixar_html_acao_resultado", baixar)
    return chamadas


class TestAcharColuna:
    def test_tolera_acento_e_espaco(self):
        colunas = {"a": "mrg. líq.", "b": "dív.brut/ patrim.", "c": "liq. corr."}
        assert _achar_coluna(colunas, "mrg. liq") == "a"
        assert _achar_coluna(colunas, "div.brut") == "b"
        assert _achar_coluna(colunas, "liq.corr", "liq. corr") == "c"

    def test_inexistente_devolve_none(self):
        assert _achar_coluna({"a": "papel"}, "roic") is None


class TestTabelaAcoes:
    def test_parseia_todos_os_campos(self, html_falso):
        tabela = buscar_todas_acoes_fundamentus()
        assert set(tabela) == {"VALE3", "BBAS3", "POMO4"}  # espaço e minúscula normalizados
        v = tabela["VALE3"]
        assert isinstance(v, FundamentosAcao)
        assert v.cotacao == pytest.approx(71.08)
        assert v.pl == pytest.approx(8.5) and v.pvp == pytest.approx(1.4)
        assert v.roe == pytest.approx(0.178) and v.roic == pytest.approx(0.152)
        assert v.margem_liquida == pytest.approx(0.185)
        assert v.divida_bruta_patrimonio == pytest.approx(0.45)
        assert v.liquidez_corrente == pytest.approx(1.35)

    def test_banco_traco_vira_none_e_nao_zero(self, html_falso):
        # Regressão (Seção 4.2.2 do Relatório): '-' para bancos não pode
        # virar 0,00 — senão bonifica indevidamente o baixo endividamento.
        b = buscar_todas_acoes_fundamentus()["BBAS3"]
        assert b.divida_bruta_patrimonio is None
        assert b.liquidez_corrente is None
        assert b.roic is None

    def test_linha_sem_papel_e_ignorada(self, monkeypatch):
        html = HTML_ACOES.replace("<td>VALE3</td>", "<td></td>")
        monkeypatch.setattr(fdm, "_baixar_html_acao_resultado", lambda: html)
        assert "VALE3" not in buscar_todas_acoes_fundamentus()


class TestBuscarFundamentosAcao:
    def test_aceita_sufixo_sa_e_minusculas(self, html_falso):
        assert buscar_fundamentos_acao("vale3.sa").ticker == "VALE3"

    def test_cache_em_memoria_uma_chamada_de_rede(self, html_falso):
        buscar_fundamentos_acao("VALE3.SA")
        buscar_fundamentos_acao("POMO4.SA")
        buscar_fundamentos_acao("BBAS3")
        assert len(html_falso) == 1

    def test_ticker_inexistente(self, html_falso):
        assert buscar_fundamentos_acao("XXXX3.SA") is None

    def test_falha_de_rede_devolve_none(self, monkeypatch):
        def falha():
            raise OSError("sem rede")
        monkeypatch.setattr(fdm, "_baixar_html_acao_resultado", falha)
        assert buscar_fundamentos_acao("VALE3.SA") is None


class _FonteFalsa:
    def __init__(self, info=None):
        self._info = info or {}

    def baixar_precos(self, ticker, periodo):
        rng = np.random.default_rng(7)
        datas = pd.bdate_range("2023-01-01", periods=300)
        return pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)), index=datas)

    def baixar_info(self, ticker):
        return self._info

    def baixar_dividendos(self, ticker):
        return pd.Series(dtype=float)


class TestIntegracaoAcaoB3:
    def test_fundamentus_substitui_yahoo(self, monkeypatch):
        fake = FundamentosAcao(ticker="VALE3", cotacao=71.0, pl=8.5, pvp=1.4, roe=0.178, roic=0.152,
                               margem_liquida=0.185, divida_bruta_patrimonio=0.45, liquidez_corrente=1.35)
        monkeypatch.setattr(fdm, "buscar_fundamentos_acao", lambda t: fake)
        r = analisar_ativo("VALE3.SA", "acao", "2y", _FonteFalsa(info={"trailingPE": 99.0, "priceToBook": 9.0}))
        assert r.fund["pl"] == 8.5 and r.fund["pvp"] == 1.4 and r.fund["roe"] == pytest.approx(0.178)
        assert r.fund["divida_bruta_patrimonio"] == 0.45 and r.fund["liquidez_corrente"] == 1.35
        assert r.fund["roic"] == pytest.approx(0.152) and r.fund["margem_liquida"] == pytest.approx(0.185)

    def test_campo_ausente_no_fundamentus_mantem_yahoo(self, monkeypatch):
        fake = FundamentosAcao(ticker="X", cotacao=1.0, pl=None, pvp=None, roe=None, roic=None,
                               margem_liquida=None, divida_bruta_patrimonio=None, liquidez_corrente=None)
        monkeypatch.setattr(fdm, "buscar_fundamentos_acao", lambda t: fake)
        r = analisar_ativo("X.SA", "acao", "2y", _FonteFalsa(info={"trailingPE": 12.0, "priceToBook": 2.0}))
        assert r.fund["pl"] == 12.0 and r.fund["pvp"] == 2.0

    def test_fundamentus_indisponivel_segue_com_yahoo(self, monkeypatch):
        def explode(t):
            raise RuntimeError("fora do ar")
        monkeypatch.setattr(fdm, "buscar_fundamentos_acao", explode)
        r = analisar_ativo("VALE3.SA", "acao", "2y", _FonteFalsa(info={"trailingPE": 12.0}))
        assert r is not None and r.fund["pl"] == 12.0 and "roic" not in r.fund

    def test_acao_eua_nao_consulta_fundamentus(self, monkeypatch):
        def nao_deveria(t):
            raise AssertionError("Fundamentus não cobre ações dos EUA")
        monkeypatch.setattr(fdm, "buscar_fundamentos_acao", nao_deveria)
        r = analisar_ativo("AAPL", "acao_us", "2y", _FonteFalsa(info={"trailingPE": 30.0}))
        assert r is not None and r.tipo == "acao_us"

    @pytest.mark.parametrize("tipo", ["etf_br", "etf_us"])
    def test_etf_nao_consulta_fundamentus(self, monkeypatch, tipo):
        monkeypatch.setattr(fdm, "buscar_fundamentos_acao", lambda t: (_ for _ in ()).throw(AssertionError()))
        assert analisar_ativo("BOVA11.SA", tipo, "2y", _FonteFalsa()) is not None

    def test_fii_com_fundamentus_indisponivel(self, monkeypatch):
        def explode(t):
            raise RuntimeError("fora do ar")
        monkeypatch.setattr(fdm, "buscar_fundamentos_fii", explode)
        r = analisar_ativo("HGLG11.SA", "fii", "2y", _FonteFalsa(info={"priceToBook": 1.0}))
        assert r is not None and "vacancia_media" not in r.fund

    def test_historico_insuficiente_devolve_none(self):
        class Curta(_FonteFalsa):
            def baixar_precos(self, ticker, periodo):
                return pd.Series([1.0, 2.0], index=pd.bdate_range("2024-01-01", periods=2))
        assert analisar_ativo("X.SA", "acao", "2y", Curta()) is None


class TestRegressaoConversaoPandas:
    """Regressão encontrada ao reescrever estes testes (Sprint 3): com
    thousands='.'/decimal=',', o pandas convertia '0,45' em '0.45' mas o
    mantinha como texto quando a coluna tinha algum '-', e o parser lia 45.
    Sem parâmetros, '900.000' (milhar) viraria 900,0."""

    def test_coluna_com_traco_nao_multiplica_por_100(self, html_falso):
        t = buscar_todas_acoes_fundamentus()
        assert t["POMO4"].divida_bruta_patrimonio == pytest.approx(0.30)
        assert t["POMO4"].liquidez_corrente == pytest.approx(1.80)

    def test_coluna_so_com_milhar_de_um_ponto(self, monkeypatch):
        html = (HTML_ACOES.replace("2.000.000.000", "2.000").replace("900.000.000", "900.000")
                .replace("30.000.000", "30.000"))
        monkeypatch.setattr(fdm, "_baixar_html_acao_resultado", lambda: html)
        assert buscar_todas_acoes_fundamentus()["VALE3"].pl == pytest.approx(8.5)
        df = fdm._ler_tabela_como_texto(html)
        assert fdm._parse_numero_br(df.loc[1, "Liq.2meses"]) == pytest.approx(900000.0)
        assert fdm._parse_numero_br(df.loc[0, "Dív.Brut/ Patrim."]) == pytest.approx(0.45)

    def test_fii_com_traco_na_coluna(self, monkeypatch):
        from test_fundamentus import HTML_AMOSTRA
        html = HTML_AMOSTRA.replace("<td>0,97</td>", "<td>-</td>")
        monkeypatch.setattr(fdm, "_baixar_html_fii_resultado", lambda: html)
        fiis = fdm.buscar_todos_fiis_fundamentus()
        assert fiis["BTLG11"].pvp is None
        assert fiis["HGLG11"].pvp == pytest.approx(1.02)
        assert fiis["KNIP11"].pvp == pytest.approx(0.88)
