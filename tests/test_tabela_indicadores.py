import pandas as pd

from carteira_analise.tabela_indicadores import _numero, meses_de_pagamento, montar_tabelas


def test_numero():
    assert _numero("6,1") == 6.1 and _numero("11,5%") == 11.5 and _numero("—") is None
    assert _numero("1.234,5") == 1234.5 and _numero("−2,0%") == -2.0


def test_meses_de_pagamento():
    hoje = pd.Timestamp("2026-11-23")
    semestral = pd.Series(1.0, index=pd.to_datetime(["2025-02-10", "2025-08-10", "2026-02-10", "2026-08-10"]))
    assert meses_de_pagamento(semestral, hoje) == "Fev · Ago"
    mensal = pd.Series(1.0, index=pd.date_range("2025-11-30", periods=12, freq="ME"))
    assert meses_de_pagamento(mensal, hoje) == "Mensal"
    assert meses_de_pagamento(None, hoje) == "—"


def test_montar_tabelas_por_grupo():
    acao = {"ticker": "CPFE3.SA", "mercado": "B3", "grupo": "Ações", "situacao": "fora", "preco_atual": 44.28,
            "variacao": 0.01,
            "contexto": {"indicadores": [{"nome": "P/L", "valor": "8,8", "leitura": "em linha com o segmento (8,6)"},
                                         {"nome": "Dividend yield 12m", "valor": "8,4%", "leitura": "x"}],
                         "preco_teto": {"teto": 62.19}, "regularidade": {"selo": "regular"}}}
    fii = {"ticker": "HGCR11.SA", "mercado": "B3", "grupo": "FIIs", "situacao": "queda", "preco_atual": 84.88,
           "variacao": -0.22, "contexto": {"indicadores": [{"nome": "Yield 12 meses", "valor": "13,8%"}],
                                           "preco_atrativo": {"pvp": 0.87, "pvp_medio_24m": 0.98, "nivel": "desconto"}}}
    etf = {"ticker": "IVV", "mercado": "EUA", "grupo": "ETF", "situacao": "realizacao", "preco_atual": 700.0,
           "variacao": 0.15, "contexto": {"indicadores": [{"nome": "Tendência (MM200)", "valor": "acima da média"}]}}
    t = montar_tabelas([acao, fii, etf], {"CPFE3.SA": {"roe": 0.2656, "ev_ebitda": 6.34, "pvp": 1.5}},
                       {}, pd.Timestamp("2026-11-23"))
    a = t["Ações B3"].loc["CPFE3.SA"]
    assert a["P/L"] == 8.8 and a["DY 12m (%)"] == 8.4 and round(a["ROE (%)"], 2) == 26.56
    assert round(a["Preço × teto (%)"]) == -29 and a["Dividendos"] == "Regular" and a["Faixa"] == "↔️ Fora"
    f = t["FIIs"].loc["HGCR11.SA"]
    assert f["P/VP"] == 0.87 and f["Preço atrativo?"] == "Com desconto" and f["DY 12m (%)"] == 13.8
    assert t["EUA"].loc["IVV", "Tipo"] == "ETF" and "ETFs B3" not in t
