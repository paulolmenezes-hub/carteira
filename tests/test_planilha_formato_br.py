"""Células em formato brasileiro (texto) na planilha — caso POMO4: preços
'6,20' digitados como texto faziam as linhas da aba Operações serem
ignoradas, e o card mostrava só a posição inicial (200 cotas a R$ 6,31)."""
import math

import pandas as pd
import pytest

from carteira_analise.planilha import (
    ler_data,
    ler_numero,
    normalizar_aba_operacoes,
    normalizar_aba_resumo,
)
from carteira_analise.posicoes import operacoes_por_ticker
from carteira_analise.regra_posicao import estado_a_partir_de_operacoes


@pytest.mark.parametrize("bruto, esperado", [
    (6.2, 6.2), (200, 200.0), ("6,20", 6.2), ("R$ 4,33", 4.33), ("1.000", 1000.0),
    ("1.234,56", 1234.56), ("4.33", 4.33), (" 300 ", 300.0), ("-50", -50.0), ("US$ 1,5", 1.5),
])
def test_ler_numero_formatos(bruto, esperado):
    assert ler_numero(bruto) == pytest.approx(esperado)


def test_ler_numero_vazio_e_texto():
    assert math.isnan(ler_numero(""))
    with pytest.raises(ValueError):
        ler_numero("abc")


@pytest.mark.parametrize("bruto, esperado", [
    ("03/07/2024", "2024-07-03"),  # dia primeiro (pd.Timestamp leria 7 de março)
    ("25/06/24", "2024-06-25"), ("2024-06-25", "2024-06-25"),
    (pd.Timestamp("2026-10-26"), "2026-10-26"), (45468, "2024-06-25"),  # número de série do Excel
])
def test_ler_data_formatos(bruto, esperado):
    assert ler_data(bruto) == pd.Timestamp(esperado)


def test_ler_data_ilegivel():
    for bruto in ("ontem", float("nan"), 12):
        with pytest.raises(ValueError):
            ler_data(bruto)


def _pomo4_texto():
    linhas = [("Compra", "300", "6,20", "03/07/2024"), ("Compra", "100", "6,14", "11/04/2025"),
              ("Compra", "200", "6,91", "02/05/2025"), ("Compra", "100", "7,20", "02/06/2025"),
              ("Bonificação", "90", "6,00", "26/12/2025"), ("Compra", "110", "6,23", "08/04/2026"),
              ("Compra", "600", "5,93", "15/06/2026"), ("Compra", "700", "4,33", "13/08/2026"),
              ("Compra", "40", "4,33", "13/08/2026"), ("Compra", "1.000", "4,11", "26/10/2026")]
    ops = pd.DataFrame([("POMO4", *l) for l in linhas],
                       columns=["Ticker", "Tipo", "Quantidade", "Preço", "Data"])
    resumo = pd.DataFrame([["POMO4", 200, "6,31", "25/06/2024"]],
                          columns=["Ticker", "Quantidade", "Preço médio", "Data início"])
    return normalizar_aba_resumo(resumo), normalizar_aba_operacoes(ops)


def test_pomo4_com_celulas_em_texto_entra_inteiro_no_preco_medio():
    resumo, ops = _pomo4_texto()
    lista = operacoes_por_ticker(resumo, ops)["POMO4.SA"]
    assert len(lista) == 11
    estado = estado_a_partir_de_operacoes(lista)
    assert estado["quantidade"] == pytest.approx(3440)
    custo = 200 * 6.31 + 300 * 6.2 + 100 * 6.14 + 200 * 6.91 + 100 * 7.2 + 90 * 6 + 110 * 6.23 \
        + 600 * 5.93 + 740 * 4.33 + 1000 * 4.11
    assert estado["preco_medio"] == pytest.approx(custo / 3440)


def test_celula_ilegivel_continua_aparecendo_no_aviso():
    ops = normalizar_aba_operacoes(pd.DataFrame(
        [["POMO4", "Compra", 10, "seis", "ontem"]], columns=["ticker", "tipo", "quantidade", "preco", "data"]))
    assert ops.loc[0, "preco"] == "seis" and ops.loc[0, "data"] == "ontem"
