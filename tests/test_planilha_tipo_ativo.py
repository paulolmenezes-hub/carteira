"""Coluna opcional de tipo de ativo nas abas Resumo e Operações."""
import pandas as pd
import pytest

from carteira_analise.planilha import (
    normalizar_aba_operacoes,
    normalizar_aba_resumo,
    normalizar_tipo_ativo,
    tipos_da_planilha,
)


@pytest.mark.parametrize("valor, ticker, esperado", [
    ("Ação", "PETR4.SA", "acao"), ("ação", "AAPL", "acao_us"), ("FII", "KNRI11", "fii"),
    ("Fundo Imobiliário", "MXRF11.SA", "fii"), ("ETF", "BOVA11.SA", "etf_br"), ("ETF", "IVV", "etf_us"),
    ("Ação (EUA)", "MSFT", "acao_us"), ("Ações", "ITSA4.SA", "acao"), (" fii ", "HGLG11.SA", "fii"),
    ("BDR", "AAPL34.SA", None), (None, "X", None), (float("nan"), "X", None),
])
def test_normalizar_tipo_ativo(valor, ticker, esperado):
    assert normalizar_tipo_ativo(valor, ticker) == esperado


def _abas(cab_resumo="Tipo de ativo", cab_ops="Tipo Ativo"):
    resumo = pd.DataFrame([["POMO4", 200, 6.31, "25/06/2024", "Ação"],
                           ["KNRI11", 10, 150.0, "01/01/2024", "FII"]],
                          columns=["Ticker", "Quantidade", "Preço médio", "Data início", cab_resumo])
    ops = pd.DataFrame([["KLBN11.SA", "Compra", 200, 18.58, "31/07/2025", "Ação"],
                        ["BOVA11.SA", "Compra", 5, 120.0, "01/02/2025", "ETF"],
                        ["IVV", "Compra", 1, 500.0, "01/02/2025", "ETF"]],
                       columns=["Ticker", "Tipo", "Quantidade", "Preço", "Data", cab_ops])
    return normalizar_aba_resumo(resumo), normalizar_aba_operacoes(ops)


def test_tipos_lidos_pelo_cabecalho():
    resumo, ops = _abas()
    assert "tipo_ativo" in resumo.columns and "tipo_ativo" in ops.columns
    tipos, avisos = tipos_da_planilha(resumo, ops)
    assert tipos == {"POMO4.SA": "acao", "KNRI11.SA": "fii", "KLBN11.SA": "acao",
                     "BOVA11.SA": "etf_br", "IVV": "etf_us"}
    assert avisos == []


def test_tipos_lidos_por_coluna_sem_cabecalho_conhecido():
    # cabeçalho repetido "Tipo" vira "Tipo.1" no pandas; reconhecida pelos valores
    resumo, ops = _abas(cab_resumo="Coluna E", cab_ops="Tipo.1")
    tipos, _ = tipos_da_planilha(resumo, ops)
    assert tipos["KNRI11.SA"] == "fii" and tipos["BOVA11.SA"] == "etf_br"


def test_sem_coluna_de_tipo_mantem_formato_antigo():
    resumo = pd.DataFrame([["POMO4", 200, 6.31, "25/06/2024", "anotação livre"]],
                          columns=["Ticker", "Quantidade", "Preço médio", "Data início", "Obs"])
    r = normalizar_aba_resumo(resumo)
    assert "tipo_ativo" not in r.columns
    assert tipos_da_planilha(r, None) == ({}, [])


def test_conflito_prefere_resumo_e_avisa():
    resumo = normalizar_aba_resumo(pd.DataFrame([["KNRI11", 10, 150.0, "01/01/2024", "FII"]],
                                                columns=["Ticker", "Quantidade", "PM", "Data", "Tipo de ativo"]))
    ops = normalizar_aba_operacoes(pd.DataFrame([["KNRI11", "Compra", 5, 140.0, "01/02/2025", "ETF"]],
                                                columns=["Ticker", "Tipo", "Quantidade", "Preço", "Data",
                                                         "Tipo de ativo"]))
    tipos, avisos = tipos_da_planilha(resumo, ops)
    assert tipos["KNRI11.SA"] == "fii"
    assert len(avisos) == 1 and "KNRI11.SA" in avisos[0]
