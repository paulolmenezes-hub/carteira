import pandas as pd

from carteira_analise.informe_b3 import (ano_do_nome, conciliar, ler_proventos_informe, resumo_conciliacao,
                                         ticker_base)


def _informe():
    df = pd.DataFrame({
        "Produto": ["KLBN11", "KLBN11", "BBSE3", "PETR4", "VIVT3", "CPTS11", "CPTS11B", "EGIE3", None, None],
        "Tipo de Evento": ["Dividendo", "Juros Sobre Capital Próprio", "Dividendo", "Rendimento",
                           "Restituição de Capital", "Rendimento", "Rendimento", "Dividendo", None, None],
        "Valor líquido": [355.46, 65.11, 1561.84, 9.70, 246.67, 630.08, 65.70, 419.18, None, "Total"],
    })
    return {"Posição - Ações": pd.DataFrame({"x": [1]}), "Proventos Recebidos": df}


def test_ler_informe():
    inf = ler_proventos_informe(_informe())
    assert len(inf) == 8
    assert set(inf["categoria"]) == {"dividendos", "jcp", "rendimentos", "restituicao"}
    assert inf[inf["produto"] == "CPTS11B"].iloc[0]["ticker"] == "CPTS11"
    assert ler_proventos_informe({"Posição": pd.DataFrame({"a": [1]})}) is None


def test_ticker_base_e_ano():
    assert ticker_base("CPTS11B") == ("CPTS11", True)
    assert ticker_base("KLBN11") == ("KLBN11", False)
    assert ticker_base("PETR4") == ("PETR4", False)
    assert ano_do_nome("relatorio-consolidado-anual-2025.xlsx") == 2025 and ano_do_nome("x.xlsx") is None


def test_conciliar():
    inf = ler_proventos_informe(_informe())
    painel = pd.DataFrame([
        {"ticker": "KLBN11.SA", "dividendos": 256.84, "jcp_bruto": 0, "ir_retido": 0, "jcp_liquido": 0,
         "rendimentos": 0, "amortizacao": 0, "total_liquido": 256.84},
        {"ticker": "BBSE3.SA", "dividendos": 1561.85, "jcp_bruto": 0, "ir_retido": 0, "jcp_liquido": 0,
         "rendimentos": 0, "amortizacao": 0, "total_liquido": 1561.85},
        {"ticker": "CPTS11.SA", "dividendos": 0, "jcp_bruto": 0, "ir_retido": 0, "jcp_liquido": 0,
         "rendimentos": 695.78, "amortizacao": 0, "total_liquido": 695.78},
        {"ticker": "HGLG11.SA", "dividendos": 0, "jcp_bruto": 0, "ir_retido": 0, "jcp_liquido": 0,
         "rendimentos": 528.0, "amortizacao": 0, "total_liquido": 528.0},
    ])
    c = conciliar(inf, painel, {"KLBN11", "BBSE3", "CPTS11", "HGLG11", "PETR4", "VIVT3"}, {"CPTS11", "HGLG11"})
    linha = lambda t, cat: c[(c["ticker"] == t) & (c["categoria"] == cat)].iloc[0]
    assert linha("BBSE3", "dividendos")["situacao"] == "ok"  # 1 centavo de diferença
    assert linha("KLBN11", "dividendos")["situacao"] == "diferente"
    assert linha("KLBN11", "dividendos")["diagnostico"].startswith("faltou no painel")
    assert linha("KLBN11", "jcp")["b3"] == 65.11 and linha("KLBN11", "jcp")["painel"] == 0
    cp = linha("CPTS11", "rendimentos")  # 630,08 + 65,70 do recibo = 695,78
    assert cp["situacao"] == "ok" and "CPTS11B" in cp["diagnostico"]
    assert linha("EGIE3", "dividendos")["situacao"] == "fora"
    assert linha("HGLG11", "rendimentos")["diagnostico"].startswith("não aparece no informe")
    assert linha("EGIE3", "dividendos")["diagnostico"] == "não está na planilha"
    assert linha("PETR4", "atualizacao")["situacao"] == "informativo"
    assert "reduz o custo" in linha("VIVT3", "restituicao")["diagnostico"]
    r = resumo_conciliacao(c)
    assert abs(r["totais"]["dividendos"]["b3"] - (355.46 + 1561.84 + 419.18)) < 1e-9
    assert r["contagem"]["ok"] == 2
    assert abs(r["totais"]["dividendos"]["b3_planilha"] - (355.46 + 1561.84)) < 1e-9  # sem a EGIE3
