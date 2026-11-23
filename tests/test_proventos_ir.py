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


def test_fii_rendimento_e_amortizacao():
    from carteira_analise.fontes.fundamentus import proventos_fii_detalhe_de_html
    html = ("<table><thead><tr><th>Última Data Com</th><th>Tipo</th><th>Data de Pagamento</th><th>Valor</th></tr>"
            "</thead><tbody><tr><td>31/01/2025</td><td>Rendimento</td><td>14/02/2025</td><td>1,10</td></tr>"
            "<tr><td>31/01/2025</td><td>Amortização</td><td>14/02/2025</td><td>0,50</td></tr>"
            "<tr><td>31/12/2025</td><td>Rendimento</td><td>15/01/2026</td><td>1,00</td></tr></tbody></table>")
    det = proventos_fii_detalhe_de_html(html)
    assert list(det["tipo"]) == ["rendimento", "amortizacao", "rendimento"]
    rec = proventos_recebidos("HGLG11.SA", [_op("2024-01-10", "compra", 10)], det)
    r25 = resumo_por_ano(rec, 2025).iloc[0]
    assert r25["rendimentos"] == pytest.approx(11) and r25["amortizacao"] == pytest.approx(5)
    assert r25["total_liquido"] == pytest.approx(16) and r25["ir_retido"] == 0
    assert resumo_por_ano(rec, 2026).iloc[0]["rendimentos"] == pytest.approx(10)  # data-com dez, pago jan


def test_a_receber_ignora_sem_data_antigo():
    from carteira_analise.proventos_ir import sem_data_antigos
    det = pd.DataFrame({"data_com": pd.to_datetime(["2023-05-02", "2026-10-01"]), "valor": [1.0, 1.0],
                        "tipo": ["jcp", "jcp"], "pagamento": [pd.NaT, pd.NaT]})
    rec = proventos_recebidos("CMIG4.SA", [_op("2023-01-02", "compra", 100)], det)
    assert list(a_receber(rec, "2026-11-23")["data_com"].dt.year) == [2026]
    assert list(sem_data_antigos(rec, "2026-11-23")["data_com"].dt.year) == [2023]


def test_bdr():
    from carteira_analise.proventos_ir import eh_bdr
    assert eh_bdr("NVDC34.SA") and not eh_bdr("PETR4.SA") and not eh_bdr("KLBN11.SA")
