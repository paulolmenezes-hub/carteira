from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from carteira_analise.posicoes import (
    fundamentos_de_resultado,
    gerar_posicoes,
    operacoes_por_ticker,
    renda_12m_por_cota,
)

DATAS = pd.bdate_range(end="2026-09-22", periods=800)


def _serie(f):
    return pd.Series([f(i) for i in range(len(DATAS))], index=DATAS, dtype=float)


class FonteFalsa:
    def __init__(self, precos, dividendos=None, falhar=()):
        self.precos, self.dividendos, self.falhar = precos, dividendos or {}, set(falhar)

    def baixar_precos(self, ticker, periodo):
        if ticker in self.falhar:
            raise RuntimeError("rede")
        return self.precos.get(ticker, pd.Series(dtype=float))

    def baixar_dividendos(self, ticker):
        if ticker in self.falhar:
            raise RuntimeError("rede")
        return self.dividendos.get(ticker, pd.Series(dtype=float))


def _resumo(linhas):
    return pd.DataFrame(linhas, columns=["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"])


def _ops(linhas):
    return pd.DataFrame(linhas, columns=["ticker", "tipo", "quantidade", "preco", "data"])


MENSAL = pd.date_range(end="2026-09-01", periods=36, freq="MS")


def test_operacoes_por_ticker_combina_abas_e_ignora_invalidas():
    resumo = _resumo([["HGLG11", 40, 100.0, None, "2024-01-02"],
                      ["XXXX3", 10, None, None, "2024-01-02"],      # sem preço: ignorada
                      ["BBAS3.SA", 10, None, 300.0, "2024-01-02"]])  # via valor investido
    ops = _ops([["HGLG11", "compra", 5, 90.0, "2024-05-01"], ["HGLG11", "xx", 1, 1.0, "2024-05-01"],
                ["HGLG11", "venda", "abc", 1.0, "2024-06-01"]])
    r = operacoes_por_ticker(resumo, ops)
    assert set(r) == {"HGLG11.SA", "BBAS3.SA"}
    assert [o["tipo"] for o in r["HGLG11.SA"]] == ["compra", "compra"]
    assert r["BBAS3.SA"][0]["preco"] == 30.0
    assert operacoes_por_ticker(None, None) == {}
    assert operacoes_por_ticker(None, _ops([])) == {}


def test_renda_12m():
    d = pd.Series(1.0, index=MENSAL)
    assert renda_12m_por_cota(d, "2026-09-22") == pytest.approx(12.0)
    d_tz = d.copy(); d_tz.index = d_tz.index.tz_localize("UTC")
    assert renda_12m_por_cota(d_tz, "2026-09-22") == pytest.approx(12.0)
    assert renda_12m_por_cota(pd.Series(dtype=float), "2026-09-22") is None
    assert renda_12m_por_cota(None, "2026-09-22") is None
    assert renda_12m_por_cota(d, "2030-01-01") is None


def test_fundamentos_de_resultado_formatos():
    @dataclass
    class F:
        pvp: float
    class R:
        pass
    r = R(); r.fund = {"pvp": 1.2}
    assert fundamentos_de_resultado(r) == {"pvp": 1.2}
    r = R(); r.fund = F(0.9)
    assert fundamentos_de_resultado(r) == {"pvp": 0.9}
    r = R(); o = R(); o.roe = 0.1; r.fundamentos = o
    assert fundamentos_de_resultado(r) == {"roe": 0.1}
    r = R(); r.fund = 5
    assert fundamentos_de_resultado(r) == {}
    assert fundamentos_de_resultado(None) == {}
    assert fundamentos_de_resultado(R()) == {}


def _cenario():
    precos = {
        "HGLG11.SA": _serie(lambda i: 100 + 30 * i / 799),  # +30% => vender 10%
        "VALE3.SA": _serie(lambda i: 80 - 20 * i / 799),    # -25% => comprar 25%, mas filtro bloqueia
        "BOVA11.SA": _serie(lambda i: 100 - 30 * i / 799),  # -30% => comprar (ETF, sem filtro)
        "QQQ": _serie(lambda i: 104.0),
        "FECH3.SA": _serie(lambda i: 10.0),
    }
    precos["QQQ"].index = precos["QQQ"].index.tz_localize("America/New_York")
    divs = {"HGLG11.SA": pd.Series(1.0, index=MENSAL),
            "VALE3.SA": pd.Series([2.0] * 24 + [0.5] * 12, index=MENSAL)}
    resumo = _resumo([[t, q, pm, None, "2023-10-02"] for t, q, pm in
                      [("HGLG11", 40, 100.0), ("VALE3", 100, 80.0), ("BOVA11", 50, 100.0),
                       ("QQQ", 3, 100.0), ("FECH3", 10, 10.0), ("SEMP3", 5, 10.0), ("ERRO3", 5, 10.0)]])
    ops = _ops([["FECH3", "venda", 10, 10.0, "2024-01-10"]])
    tipos = {"HGLG11.SA": "fii", "VALE3.SA": "acao", "BOVA11.SA": "etf_br", "QQQ": "etf_us"}
    return FonteFalsa(precos, divs, falhar={"ERRO3.SA"}), resumo, ops, tipos


def test_gerar_posicoes_cenario_completo():
    fonte, resumo, ops, tipos = _cenario()
    fund = {"HGLG11.SA": {"pvp": 0.95, "qtd_imoveis": 10, "segmento": "Logística", "vacancia_media": 0.03},
            "VALE3.SA": {"roe": 0.15, "divida_bruta_patrimonio": 0.4, "liquidez_corrente": 1.4}}
    cards, avisos = gerar_posicoes(resumo, ops, fonte, tipos, fund)
    por = {c["ticker"]: c for c in cards}
    assert set(por) == {"HGLG11.SA", "VALE3.SA", "BOVA11.SA", "QQQ"}  # FECH3 encerrada
    assert por["HGLG11.SA"]["acao"] == "vender" and por["HGLG11.SA"]["posicao"] == "💰 Vender 4 cota(s)"
    assert any("por mês" in l for l in por["HGLG11.SA"]["linhas"])
    assert por["VALE3.SA"]["acao"] == "manter_nao_aumente" and "rendimentos" in por["VALE3.SA"]["linhas"][0]
    assert por["BOVA11.SA"]["acao"] == "comprar"
    assert por["QQQ"]["acao"] == "manter" and "US$" in por["QQQ"]["titulo"]
    assert [c["prioridade"] for c in cards] == sorted(c["prioridade"] for c in cards)
    assert any("SEMP3.SA" in a for a in avisos) and any("ERRO3.SA" in a for a in avisos)


def test_fundamento_ruim_bloqueia_compra():
    fonte, resumo, ops, tipos = _cenario()
    fonte.dividendos.pop("VALE3.SA")
    cards, _ = gerar_posicoes(resumo, ops, fonte, tipos, {"VALE3.SA": {"roe": -0.1}})
    vale = next(c for c in cards if c["ticker"] == "VALE3.SA")
    assert vale["acao"] == "manter_nao_aumente" and "prejuízo" in vale["linhas"][0]
    cards, _ = gerar_posicoes(resumo, ops, fonte, tipos, {"VALE3.SA": {"roe": 0.2}})
    assert next(c for c in cards if c["ticker"] == "VALE3.SA")["acao"] == "comprar"


def test_tipo_nao_confirmado_gera_aviso():
    fonte, resumo, ops, _ = _cenario()
    cards, avisos = gerar_posicoes(resumo, ops, fonte, tipos=None)
    assert any("BOVA11.SA" in a and "não confirmado" in a for a in avisos)
    assert next(c for c in cards if c["ticker"] == "BOVA11.SA")["acao"] == "comprar"


def test_operacao_invalida_vira_aviso():
    fonte, _, _, tipos = _cenario()
    resumo = _resumo([["HGLG11", 40, 100.0, None, "2023-10-02"]])
    ops = _ops([["HGLG11", "venda", 1, None, "2024-01-10"]])  # preço ausente: linha ignorada
    cards, avisos = gerar_posicoes(resumo, ops, fonte, tipos)
    assert len(cards) == 1


def test_estado_com_erro_vira_aviso(monkeypatch):
    import carteira_analise.posicoes as p
    fonte, resumo, ops, tipos = _cenario()
    def explode(*a, **k):
        raise KeyError("x")
    monkeypatch.setattr(p, "estado_a_partir_de_operacoes", explode)
    cards, avisos = gerar_posicoes(resumo, ops, fonte, tipos)
    assert cards == [] and any("operações inválidas" in a for a in avisos)
