"""Leitura da planilha (abas Resumo + Operações), combinação das duas abas
por ativo e montagem do Dashboard da Carteira — tudo com fonte de dados
falsa, sem rede."""
from datetime import date

import pandas as pd
import pytest

from carteira_analise.planilha import (
    LinhaCarteira,
    achar_aba,
    construir_dashboard_por_ativo,
    normalizar_aba_operacoes,
    normalizar_aba_resumo,
    normalizar_ticker,
    processar_carteira_combinada,
)


class FonteFalsa:
    def __init__(self, precos=None, dividendos=None, falhar_preco=(), falhar_div=()):
        self.precos = precos or {}
        self.dividendos = dividendos or {}
        self.falhar_preco, self.falhar_div = set(falhar_preco), set(falhar_div)

    def baixar_precos(self, ticker, periodo):
        if ticker in self.falhar_preco:
            raise RuntimeError("rede")
        valor = self.precos.get(ticker, 10.0)
        return pd.Series([valor * 0.9, valor], index=pd.bdate_range("2026-09-21", periods=2))

    def baixar_dividendos(self, ticker):
        if ticker in self.falhar_div:
            raise RuntimeError("rede")
        return self.dividendos.get(ticker, pd.Series(dtype=float))


def resumo(linhas):
    return pd.DataFrame(linhas, columns=["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"])


def operacoes(linhas):
    return pd.DataFrame(linhas, columns=["ticker", "tipo", "quantidade", "preco", "data"])


# ---------------------------------------------------------------- tickers
@pytest.mark.parametrize("bruto,esperado,corrigido", [
    ("POMO4", "POMO4.SA", True),
    (" hglg11 ", "HGLG11.SA", True),
    ("PETR4.SA", "PETR4.SA", False),
    ("BRK-B", "BRK-B", False),
    ("AAPL", "AAPL", False),
    ("QQQ", "QQQ", False),
    ("ABC", "ABC", False),
])
def test_normalizar_ticker(bruto, esperado, corrigido):
    assert normalizar_ticker(bruto) == (esperado, corrigido)


# ---------------------------------------------------------------- aba Resumo
def test_resumo_aceita_sinonimos_e_acentos():
    df = pd.DataFrame({"Ativo": ["POMO4"], "Qtde": [100], "Preço Médio": [6.5], "Data Início": ["2024-01-02"]})
    r = normalizar_aba_resumo(df)
    assert list(r.columns) == ["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"]
    assert r.loc[0, "preco_medio"] == 6.5 and r.loc[0, "valor_investido"] is None


def test_resumo_com_valor_investido_no_lugar_do_preco():
    df = pd.DataFrame({"ticker": ["X"], "quantidade": [10], "valor_investido": [100.0], "data_inicio": ["2024-01-02"]})
    r = normalizar_aba_resumo(df)
    assert r.loc[0, "valor_investido"] == 100.0 and r.loc[0, "preco_medio"] is None


def test_resumo_sem_coluna_obrigatoria():
    with pytest.raises(ValueError, match="data_inicio"):
        normalizar_aba_resumo(pd.DataFrame({"ticker": ["X"], "quantidade": [1], "preco_medio": [1.0]}))


def test_resumo_sem_preco_nem_valor():
    with pytest.raises(ValueError, match="preco_medio"):
        normalizar_aba_resumo(pd.DataFrame({"ticker": ["X"], "quantidade": [1], "data": ["2024-01-02"]}))


# ---------------------------------------------------------------- aba Operações
def test_operacoes_aceita_sinonimos():
    df = pd.DataFrame({"Papel": ["X"], "Operação": ["compra"], "Qtd": [5], "Preço Unitário": [10.0],
                       "Data da Operação": ["2024-01-02"], "Obs": ["ignorada"]})
    r = normalizar_aba_operacoes(df)
    assert list(r.columns) == ["ticker", "tipo", "quantidade", "preco", "data"]


def test_operacoes_sem_coluna_obrigatoria():
    with pytest.raises(ValueError, match="preco"):
        normalizar_aba_operacoes(pd.DataFrame({"ticker": ["X"], "tipo": ["compra"], "quantidade": [1], "data": ["x"]}))


# ---------------------------------------------------------------- achar_aba
def test_achar_aba_tolera_maiusculas_e_acentos():
    abas = {"OPERAÇÕES": pd.DataFrame({"a": [1]}), "Resumo da Carteira": pd.DataFrame({"b": [2]})}
    assert list(achar_aba(abas, ["Operacoes"]).columns) == ["a"]
    assert list(achar_aba(abas, ["Resumo", "Resumo da carteira"]).columns) == ["b"]
    assert achar_aba(abas, ["Posição"]) is None


# ---------------------------------------------------------------- carteira combinada
def test_so_resumo_posicao_aberta():
    fonte = FonteFalsa(precos={"POMO4.SA": 8.0})
    [l] = processar_carteira_combinada(resumo([["POMO4", 100, 6.5, None, "2024-01-02"]]), None, fonte)
    assert isinstance(l, LinhaCarteira)
    assert l.situacao == "Aberta" and l.moeda == "R$" and l.origem == "posição inicial"
    assert l.quantidade_aberta == 100 and l.preco_atual == 8.0
    assert l.ganho_nao_realizado == pytest.approx(150.0)


def test_resumo_mais_operacoes_venda_parcial():
    fonte = FonteFalsa(precos={"POMO4.SA": 8.0})
    ops = operacoes([["POMO4", "venda", 40, 7.0, "2024-06-03"], ["pomo4", "compra", 10, 5.0, "2024-07-01"]])
    [l] = processar_carteira_combinada(resumo([["POMO4", 100, 6.5, None, "2024-01-02"]]), ops, fonte)
    assert l.origem == "posição inicial + 2 operação(ões)"
    assert l.quantidade_aberta == 70
    assert l.ganho_realizado == pytest.approx(40 * 0.5)


def test_so_operacoes_e_posicao_encerrada():
    fonte = FonteFalsa(precos={"AAPL": 200.0})
    ops = operacoes([["AAPL", "buy", 10, 100.0, "2024-01-02"], ["AAPL", "sell", 10, 150.0, "2024-05-02"]])
    [l] = processar_carteira_combinada(None, ops, fonte)
    assert l.situacao == "Encerrada" and l.moeda == "US$" and l.ganho_realizado == pytest.approx(500.0)


def test_resumo_via_valor_investido():
    fonte = FonteFalsa(precos={"X": 12.0})
    [l] = processar_carteira_combinada(resumo([["X", 10, None, 100.0, "2024-01-02"]]), None, fonte)
    assert l.preco_medio_atual == pytest.approx(10.0)


def test_linhas_invalidas_viram_avisos_ou_erro():
    fonte = FonteFalsa()
    ops = operacoes([["X", "transferência", 1, 1.0, "2024-01-02"],
                     ["X", "compra", "abc", 1.0, "2024-01-02"]])
    res = resumo([["X", 10, None, None, "2024-01-02"]])
    [l] = processar_carteira_combinada(res, ops, fonte)
    assert l.situacao == "erro" and l.erro == "nenhuma posição/operação válida"
    assert len(l.avisos) == 3
    assert any("tipo inválido" in a for a in l.avisos)
    assert any("ilegível" in a for a in l.avisos)
    assert any("posição inicial" in a for a in l.avisos)


def test_venda_maior_que_posicao_vira_erro():
    ops = operacoes([["X", "compra", 5, 10.0, "2024-01-02"], ["X", "venda", 10, 10.0, "2024-02-01"]])
    [l] = processar_carteira_combinada(None, ops, FonteFalsa())
    assert l.situacao == "erro" and l.erro


def test_falha_de_rede_nao_quebra():
    fonte = FonteFalsa(falhar_preco={"X"}, falhar_div={"X"})
    [l] = processar_carteira_combinada(resumo([["X", 10, 5.0, None, "2024-01-02"]]), None, fonte)
    assert l.situacao == "Aberta" and l.preco_atual is None and l.ganho_nao_realizado is None


def test_operacoes_vazias_e_varios_tickers_ordenados():
    fonte = FonteFalsa()
    res = resumo([["VALE3", 1, 1.0, None, "2024-01-02"], ["AAPL", 1, 1.0, None, "2024-01-02"]])
    linhas = processar_carteira_combinada(res, operacoes([]), fonte)
    assert [l.ticker for l in linhas] == ["AAPL", "VALE3.SA"]


def test_renda_recebida_entra_no_ganho():
    divs = pd.Series([1.0, 1.0], index=pd.to_datetime(["2024-03-01", "2024-04-01"]))
    fonte = FonteFalsa(precos={"X": 10.0}, dividendos={"X": divs})
    [l] = processar_carteira_combinada(resumo([["X", 10, 10.0, None, "2024-01-02"]]), None, fonte)
    assert l.renda_recebida == pytest.approx(20.0)
    assert l.ganho_total == pytest.approx(20.0)


# ---------------------------------------------------------------- dashboard
def test_dashboard_sem_analise_usa_so_planilha():
    res = resumo([["HGLG11", 10, 100.0, None, "2024-01-02"], ["AAPL", 5, 150.0, None, "2024-01-02"]])
    ops = operacoes([["HGLG11", "compra", 5, 90.0, "2024-02-01"], ["HGLG11", "venda", 15, 110.0, "2024-03-01"],
                     ["HGLG11", "venda", "x", 1.0, "2024-03-02"], ["HGLG11", "outro", 1, 1.0, "2024-03-03"]])
    linhas = construir_dashboard_por_ativo(res, ops, tipos={"HGLG11.SA": "fii", "AAPL": "acao_us"})
    por = {l.ticker: l for l in linhas}
    h = por["HGLG11.SA"]
    assert (h.mercado, h.tipo_grupo, h.rotulo_renda, h.moeda) == ("B3", "FIIs", "Rendimentos", "R$")
    assert h.saldo_inicial_valor == 1000.0 and h.saldo_inicial_data == date(2024, 1, 2) and h.qtde_inicial == 10
    assert (h.compras_quantidade, h.compras_valor) == (5, 450.0)
    assert (h.vendas_quantidade, h.vendas_valor) == (15, 1650.0)
    assert h.preco_medio == pytest.approx(1450 / 15)
    assert h.situacao == "Encerrada" and h.saldo_final_valor is None
    a = por["AAPL"]
    assert (a.mercado, a.tipo_grupo, a.rotulo_renda, a.situacao) == ("EUA", "Ações", "Dividendos/JCP", "Ativa")


def test_dashboard_tipo_desconhecido_e_etf():
    linhas = construir_dashboard_por_ativo(resumo([["BOVA11", 1, 100.0, None, "2024-01-02"],
                                                   ["ZZZZ3", 1, 1.0, None, "2024-01-02"]]),
                                           None, tipos={"BOVA11.SA": "etf_br"})
    por = {l.ticker: l for l in linhas}
    assert por["BOVA11.SA"].tipo_grupo == "ETF" and por["ZZZZ3.SA"].tipo_grupo == "Ações"


def test_dashboard_com_analise_completa_rentabilidade():
    fonte = FonteFalsa(precos={"POMO4.SA": 8.0})
    res = resumo([["POMO4", 100, 6.0, None, "2024-01-02"]])
    ops = operacoes([["POMO4", "venda", 50, 7.0, "2024-06-03"]])
    ganho = processar_carteira_combinada(res, ops, fonte)
    [l] = construir_dashboard_por_ativo(res, ops, linhas_ganho=ganho, data_analise=date(2026, 9, 28))
    assert l.situacao == "Ativa" and l.saldo_final_qtde == 50 and l.preco_final == 8.0
    assert l.saldo_final_valor == 400.0 and l.saldo_final_data == date(2026, 9, 28)
    assert l.ganho_realizado == pytest.approx(50.0) and l.ganho_nao_realizado == pytest.approx(100.0)
    assert l.rentabilidade_total == pytest.approx(150.0)
    assert l.rentabilidade_pct == pytest.approx(150.0 / 600.0)


def test_dashboard_com_analise_posicao_encerrada_e_sem_preco():
    res = resumo([["X", 10, 10.0, None, "2024-01-02"]])
    ops = operacoes([["X", "venda", 10, 12.0, "2024-06-03"]])
    ganho = processar_carteira_combinada(res, ops, FonteFalsa(falhar_preco={"X"}))
    [l] = construir_dashboard_por_ativo(res, ops, linhas_ganho=ganho)
    assert l.situacao == "Encerrada" and l.saldo_final_valor is None
    assert l.rentabilidade_total == pytest.approx(20.0)


def test_dashboard_resumo_invalido_nao_quebra_e_via_valor_investido():
    res = resumo([["X", "abc", 1.0, None, "2024-01-02"], ["Y", 10, None, 50.0, "2024-01-02"]])
    por = {l.ticker: l for l in construir_dashboard_por_ativo(res, None)}
    assert por["X"].saldo_inicial_valor is None and por["X"].situacao == "Encerrada"
    assert por["Y"].saldo_inicial_valor == pytest.approx(50.0) and por["Y"].preco_medio == pytest.approx(5.0)


def test_dashboard_vazio():
    assert construir_dashboard_por_ativo(None, None) == []
    assert construir_dashboard_por_ativo(None, operacoes([])) == []


# ---------------------------------------------------------------- quantidades inválidas (Sprint 4)
import pandas as _pd
import pytest as _pytest

from carteira_analise.planilha import interpretar_quantidade_operacao, processar_carteira_combinada
from carteira_analise.posicoes import operacoes_por_ticker


@_pytest.mark.parametrize("tipo,valor,esperado", [
    ("compra", 10, 10.0), ("venda", -5, 5.0), ("venda", "3", 3.0),
    ("compra", 0, None), ("venda", float("nan"), None), ("compra", -2, None), ("compra", "abc", None)])
def test_interpretar_quantidade(tipo, valor, esperado):
    q, motivo = interpretar_quantidade_operacao(tipo, valor)
    assert q == esperado and (motivo is None) == (esperado is not None)


class _FonteConstante:
    def baixar_precos(self, ticker, periodo):
        return _pd.Series([20.0, 21.0], index=_pd.bdate_range("2026-09-01", periods=2))

    def baixar_dividendos(self, ticker):
        return _pd.Series(dtype=float)


def test_linha_invalida_nao_derruba_a_carteira():
    ops = _pd.DataFrame([
        ["IVV", "compra", 10, 500.0, "2025-01-02"],
        ["IVV", "venda", -4, 550.0, "2025-06-02"],      # venda negativa: aceita
        ["IVV", "compra", 0, 520.0, "2025-07-01"],      # zero: ignorada
        ["QQQ", "compra", float("nan"), 400.0, "2025-01-02"],  # vazia: ignorada
        ["QQQ", "compra", 5, 400.0, "2025-01-03"],
    ], columns=["ticker", "tipo", "quantidade", "preco", "data"])
    linhas = processar_carteira_combinada(None, ops, _FonteConstante())
    por = {l.ticker: l for l in linhas}
    assert any("zero" in a for a in por["IVV"].avisos)
    assert any("vazia" in a for a in por["QQQ"].avisos)
    assert por["IVV"].situacao != "erro" and por["QQQ"].situacao != "erro"
    cen = operacoes_por_ticker(None, ops)
    assert [o["quantidade"] for o in cen["IVV"]] == [10.0, 4.0] and len(cen["QQQ"]) == 1


def test_bonificacao_a_preco_zero_entra_na_quantidade_sem_custo():
    """Regressão (ITSA4/POMO4): compra lançada com preço 0 (bonificação) era
    descartada no saldo final, mas somada na coluna de compras do extrato."""
    import pandas as pd
    from carteira_analise.planilha import processar_carteira_combinada, construir_dashboard_por_ativo
    from carteira_analise.posicoes import operacoes_por_ticker
    from carteira_analise.regra_posicao import estado_a_partir_de_operacoes

    resumo = pd.DataFrame({"ticker": ["ITSA4"], "quantidade": [200], "preco_medio": [9.8127],
                           "valor_investido": [None], "data_inicio": [pd.Timestamp("2024-08-07")]})
    ops = pd.DataFrame({"ticker": ["ITSA4", "ITSA4"], "tipo": ["compra", "compra"],
                        "data": [pd.Timestamp("2025-01-10"), pd.Timestamp("2025-12-22")],
                        "quantidade": [455, 33], "preco": [9.3470, 0.0]})

    class Fonte:
        def baixar_precos(self, t, p):
            return pd.Series([16.07], index=[pd.Timestamp("2026-10-09")])

        def baixar_dividendos(self, t):
            return pd.Series(dtype=float)

    linhas = processar_carteira_combinada(resumo, ops, Fonte())
    assert linhas[0].quantidade_aberta == 688
    assert linhas[0].avisos == []
    custo = 200 * 9.8127 + 455 * 9.3470
    assert abs(linhas[0].preco_medio_atual - custo / 688) < 1e-9
    dash = construir_dashboard_por_ativo(resumo, ops, linhas)
    assert dash[0].saldo_final_qtde == 688 == dash[0].qtde_inicial + dash[0].compras_quantidade + dash[0].eventos_quantidade

    estado = estado_a_partir_de_operacoes(operacoes_por_ticker(resumo, ops)["ITSA4.SA"])
    assert estado["quantidade"] == 688 and estado["n_reforcos"] == 1  # bonificação não conta como aumento


def test_venda_a_preco_zero_continua_invalida():
    from carteira_analise.ganhos import Operacao
    import datetime as dt
    import pytest
    with pytest.raises(ValueError):
        Operacao(data=dt.date(2025, 1, 1), tipo="venda", quantidade=1, preco=0)
    with pytest.raises(ValueError):
        Operacao(data=dt.date(2025, 1, 1), tipo="compra", quantidade=1, preco=-1)


def test_extrato_usa_o_mesmo_preco_medio_dos_cards_quando_aberta():
    import pandas as pd
    from carteira_analise.planilha import processar_carteira_combinada, construir_dashboard_por_ativo
    ops = pd.DataFrame({"ticker": ["X3"] * 3, "tipo": ["compra", "venda", "compra"],
                        "data": pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-01"]),
                        "quantidade": [10, 5, 5], "preco": [10.0, 12.0, 20.0]})

    class Fonte:
        def baixar_precos(self, t, p):
            return pd.Series([18.0], index=[pd.Timestamp("2024-04-01")])

        def baixar_dividendos(self, t):
            return pd.Series(dtype=float)

    linhas = processar_carteira_combinada(None, ops, Fonte())
    d = construir_dashboard_por_ativo(None, ops, linhas)[0]
    assert d.preco_medio == linhas[0].preco_medio_atual == 15.0  # (5×10 + 5×20) / 10


def test_cenarios_ignora_encerrada_sem_cotacao():
    import pandas as pd
    from carteira_analise.cenarios import gerar_cenarios
    ops = pd.DataFrame({"ticker": ["BCFF11", "BCFF11"], "tipo": ["compra", "venda"],
                        "data": pd.to_datetime(["2023-01-02", "2024-01-02"]),
                        "quantidade": [10, 10], "preco": [80.0, 85.0]})

    class Fonte:
        def baixar_precos(self, t, p):
            raise RuntimeError("sem cotação")

        def baixar_dividendos(self, t):
            return pd.Series(dtype=float)

    cards, avisos = gerar_cenarios(None, ops, Fonte())
    assert cards == [] and avisos == []


def _fonte_fixa(preco=10.0):
    import pandas as pd

    class Fonte:
        def baixar_precos(self, t, p):
            return pd.Series([preco], index=[pd.Timestamp("2026-10-01")])

        def baixar_dividendos(self, t):
            return pd.Series(dtype=float)
    return Fonte()


def _ops(linhas):
    import pandas as pd
    return pd.DataFrame(linhas, columns=["ticker", "tipo", "quantidade", "preco", "data"]).assign(
        data=lambda d: pd.to_datetime(d["data"]))


def test_desdobramento_multiplica_quantidade_e_mantem_custo():
    import pytest
    from carteira_analise.planilha import processar_carteira_combinada, construir_dashboard_por_ativo
    ops = _ops([["X3", "compra", 100, 20.0, "2024-01-02"],
                ["X3", "Desdobramento", 100, None, "2024-06-03"],   # 1:2 — recebeu 100 ações
                ["X3", "venda", 50, 12.0, "2024-07-01"]])
    l = processar_carteira_combinada(None, ops, _fonte_fixa(11.0))[0]
    assert l.quantidade_aberta == 150 and l.preco_medio_atual == pytest.approx(10.0)
    assert l.ganho_realizado == pytest.approx(50 * (12.0 - 10.0)) and l.avisos == []
    d = construir_dashboard_por_ativo(None, ops, [l])[0]
    assert d.eventos_quantidade == 100
    assert d.compras_quantidade + d.eventos_quantidade - d.vendas_quantidade == d.saldo_final_qtde


def test_grupamento_reduz_quantidade_e_mantem_custo():
    import pytest
    from carteira_analise.planilha import processar_carteira_combinada
    ops = _ops([["X3", "compra", 1000, 1.0, "2024-01-02"],
                ["X3", "grupamento", 900, None, "2024-06-03"]])     # 10:1 — sobram 100
    l = processar_carteira_combinada(None, ops, _fonte_fixa(12.0))[0]
    assert l.quantidade_aberta == pytest.approx(100) and l.preco_medio_atual == pytest.approx(10.0)
    assert l.ganho_nao_realizado == pytest.approx(200.0)


def test_bonificacao_com_custo_atribuido_entra_no_custo():
    import pytest
    from carteira_analise.planilha import processar_carteira_combinada
    from carteira_analise.posicoes import operacoes_por_ticker
    from carteira_analise.regra_posicao import estado_a_partir_de_operacoes
    ops = _ops([["ABCD3", "compra", 100, 10.0, "2024-01-02"],
                ["ABCD3", "Bonificação", 10, 5.0, "2024-06-03"]])
    l = processar_carteira_combinada(None, ops, _fonte_fixa())[0]
    assert l.quantidade_aberta == 110 and l.preco_medio_atual == pytest.approx(1050 / 110)
    estado = estado_a_partir_de_operacoes(operacoes_por_ticker(None, ops)["ABCD3.SA"])
    assert estado["n_reforcos"] == 0  # custo atribuído abaixo do PM não é "aumento na queda"


def test_evento_sem_posicao_ou_grupamento_excessivo_vira_erro():
    from carteira_analise.planilha import processar_carteira_combinada
    sem_posicao = _ops([["X3", "desdobramento", 10, None, "2024-01-02"],
                        ["X3", "compra", 10, 1.0, "2024-02-01"]])
    assert processar_carteira_combinada(None, sem_posicao, _fonte_fixa())[0].situacao == "erro"
    excessivo = _ops([["X3", "compra", 10, 1.0, "2024-01-02"], ["X3", "grupamento", 10, None, "2024-02-01"]])
    l = processar_carteira_combinada(None, excessivo, _fonte_fixa())[0]
    assert l.situacao == "erro" and "grupamento" in l.erro


def test_desdobramento_ajusta_operacoes_anteriores_para_a_escala_atual():
    import pandas as pd
    from carteira_analise.planilha import aplicar_eventos_societarios
    ops = [{"data": pd.Timestamp("2024-01-02"), "tipo": "compra", "quantidade": 10.0, "preco": 30.0},
           {"data": pd.Timestamp("2024-02-01"), "tipo": "desdobramento", "quantidade": 20.0, "preco": 0.0}]
    r = aplicar_eventos_societarios(ops)
    assert len(r) == 1 and r[0]["quantidade"] == 30 and r[0]["preco"] == 10
    assert ops[0]["quantidade"] == 10  # não altera a lista recebida


def test_tipo_invalido_lista_os_tipos_aceitos():
    from carteira_analise.planilha import processar_carteira_combinada
    ops = _ops([["X3", "compra", 10, 1.0, "2024-01-02"], ["X3", "subscrição", 1, 1.0, "2024-02-01"]])
    l = processar_carteira_combinada(None, ops, _fonte_fixa())[0]
    assert any("grupamento" in a for a in l.avisos)
