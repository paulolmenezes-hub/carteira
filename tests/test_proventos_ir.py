import pandas as pd
import pytest

from carteira_analise.fontes.fundamentus import proventos_acao_detalhe_de_html
from carteira_analise.proventos_ir import (a_receber, proventos_recebidos, quantidade_na_data_com,
                                           resumo_por_ano)

HTML = ("<table><tr><th>Data ▼▲</th><th>Valor ▼▲</th><th>Tipo ▼▲</th><th>Data de Pagamento ▼▲</th>"
        "<th>Por quantas ações ▼▲</th></tr>"
        "<tr><td>29/04/2025</td><td>2,50</td><td>DIVIDENDO</td><td>30/06/2025</td><td>1</td></tr>"
        "<tr><td>29/04/2025</td><td>0,30</td><td>JRS CAP PROPRIO</td><td>30/06/2025</td><td>1</td></tr>"
        "<tr><td>15/12/2025</td><td>1,00</td><td>JRS CAP PROPRIO</td><td>20/03/2026</td><td>1</td></tr>"
        "<tr><td>10/11/2026</td><td>100,00</td><td>DIVIDENDO</td><td>-</td><td>100</td></tr></table>")


def _op(d, tipo, q, p=10.0):
    return {"data": pd.Timestamp(d), "tipo": tipo, "quantidade": q, "preco": p}


def test_le_tipo_e_data_de_pagamento():
    df = proventos_acao_detalhe_de_html(HTML)
    assert list(df["tipo"]) == ["dividendo", "jcp", "jcp", "dividendo"]
    assert df["pagamento"].iloc[0] == pd.Timestamp("2025-06-30") and pd.isna(df["pagamento"].iloc[-1])
    assert df["valor"].iloc[-1] == pytest.approx(1.0)  # 100 por 100 ações


def test_quantidade_na_data_com_conta_compra_do_dia():
    ops = [_op("2025-01-10", "compra", 100), _op("2025-04-29", "compra", 50), _op("2025-05-02", "venda", 30)]
    assert quantidade_na_data_com(ops, "2025-04-29") == 150


def test_jcp_retem_15_e_agrupa_pelo_ano_do_pagamento():
    ops = [_op("2025-01-10", "compra", 100)]
    rec = proventos_recebidos("ITSA4.SA", ops, proventos_acao_detalhe_de_html(HTML))
    r25 = resumo_por_ano(rec, 2025).iloc[0]
    assert r25["dividendos"] == pytest.approx(250) and r25["jcp_bruto"] == pytest.approx(30)
    assert r25["ir_retido"] == pytest.approx(4.5) and r25["jcp_liquido"] == pytest.approx(25.5)
    assert r25["total_liquido"] == pytest.approx(275.5)
    # JCP com data-com em dez/2025 e pagamento em mar/2026 conta em 2026
    r26 = resumo_por_ano(rec, 2026).iloc[0]
    assert r26["jcp_bruto"] == pytest.approx(100) and r26["dividendos"] == 0
    assert len(a_receber(rec, "2026-11-23")) == 1  # o de nov/2026, sem data de pagamento


def test_provento_antes_da_primeira_compra_nao_conta():
    ops = [_op("2025-06-01", "compra", 100)]
    rec = proventos_recebidos("X", ops, proventos_acao_detalhe_de_html(HTML))
    assert set(rec["data_com"].dt.year) == {2025, 2026} and rec["data_com"].min() >= pd.Timestamp("2025-06-01")
