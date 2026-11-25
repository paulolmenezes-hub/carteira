from datetime import date

import pandas as pd

from carteira_analise.relatorio_pdf import gerar_relatorio, pct, reais, rotulo_mes


def test_formatos():
    assert reais(1234.5) == "R$ 1.234,50" and reais(-10, sinal=True) == "−R$ 10,00"
    assert reais(10, sinal=True) == "+R$ 10,00" and reais(None) == "—"
    assert pct(0.153) == "+15,3%" and pct(-0.05, 0) == "−5%" and pct(0.1, 1, False) == "10,0%"
    assert rotulo_mes(pd.Period("2026-10", "M")) == "out/26"


def test_gera_pdf_completo_e_minimo():
    meses = pd.period_range("2025-11", "2026-10", freq="M")
    serie = pd.DataFrame({"patrimonio": range(100, 112), "custo": [100] * 12,
                          "rentabilidade": [i / 100 for i in range(12)], "cdi": [i / 120 for i in range(12)]},
                         index=meses)
    d = {"data": date(2026, 11, 25), "periodo": "nov/25 a out/26",
         "resumo": {"patrimonio": 111.0, "aportes_liq": 0.0, "proventos": 5.0, "resultado": 16.0,
                    "rentabilidade": 0.11, "cdi": 0.09, "ibov": None, "tir": 0.12},
         "serie": serie,
         "anos": [{"rotulo": "2026 (10 meses)", "rentabilidade": .1, "cdi": .12, "ibov": None, "resultado": 10,
                   "proventos": 5}],
         "alocacao": [{"classe": "Ações B3", "meta": 45, "hoje": 50.0, "falta": -5.0, "leitura": "na meta"}],
         "limites": {"maximo": 10.0, "minimo": 0.0},
         "ativos": [{"ticker": "PETR4", "classe": "Ações", "peso": 12.0, "preco_medio": 30, "preco": 40,
                     "variacao": .33, "faixa": "realizacao", "dy": .1, "avisos": ["peso acima do limite"]}],
         "proventos_mes": pd.DataFrame({"Ações": [1.0] * 12}, index=meses),
         "proventos_ativos": [("PETR4", 12.0)], "destaques": ["Uma frase."]}
    pdf = gerar_relatorio(d)
    assert pdf[:4] == b"%PDF" and len(pdf) > 20000  # com gráficos
    minimo = gerar_relatorio({"resumo": {"patrimonio": 0.0, "rentabilidade": 0.0}})
    assert minimo[:4] == b"%PDF"
