import numpy as np
import pandas as pd
import pytest

from carteira_analise.fontes.fundamentus import tabela_pares_de_html
from carteira_analise.raio_x import (pares_fii, posicao_entre_pares, proventos_por_ano, proximo_provento, risco_12m,
                                     selecionar_pares)


def test_risco_12m():
    idx = pd.bdate_range("2025-11-01", "2026-11-20")
    precos = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    precos.iloc[100:120] = 90.0  # queda de ~10% a ~25%
    r = risco_12m(precos)
    assert r["maior_queda"] < -0.1 and r["volatilidade"] > 0
    assert risco_12m(precos.head(10)) is None


def test_pares_e_posicao():
    html = ("<table><thead><tr><th>Papel</th><th>Cotação</th><th>P/L</th><th>P/VP</th><th>Div.Yield</th>"
            "<th>ROE</th><th>Liq.2meses</th></tr></thead><tbody>"
            "<tr><td>CPFE3</td><td>43,35</td><td>8,5</td><td>2,11</td><td>8,6%</td><td>24,7%</td><td>50.000.000</td></tr>"
            "<tr><td>EGIE3</td><td>40</td><td>10,2</td><td>2,9</td><td>7,4%</td><td>27,1%</td><td>80.000.000</td></tr>"
            "<tr><td>CMIG4</td><td>10</td><td>5,7</td><td>1,0</td><td>11,7%</td><td>17,9%</td><td>200.000.000</td></tr>"
            "<tr><td>XXXX3</td><td>1</td><td>-3,0</td><td>0,5</td><td>0%</td><td>-5%</td><td>2.000.000</td></tr>"
            "<tr><td>CPFE4</td><td>43</td><td>8,4</td><td>2,1</td><td>8,6%</td><td>24,7%</td><td>90.000.000</td></tr>"
            "<tr><td>YYYY3</td><td>1</td><td>9,0</td><td>1,0</td><td>5%</td><td>10%</td><td>10.000</td></tr>"
            "</tbody></table>")
    t = tabela_pares_de_html(html)
    assert t.loc[t.papel == "CPFE3", "dy"].iloc[0] == pytest.approx(0.086)
    pares = selecionar_pares(t, "CPFE3.SA")
    # com prejuízo entra (P/L negativo); pouca liquidez e outra classe da mesma empresa (CPFE4) não
    assert list(pares["papel"]) == ["CMIG4", "EGIE3", "CPFE3", "XXXX3"]
    assert posicao_entre_pares(pares) == "2º menor P/L e 2º maior DY entre 4"


def test_proventos_por_ano_e_proximo():
    det = pd.DataFrame({"data_com": pd.to_datetime(["2025-04-29", "2025-04-29", "2026-04-29", "2026-11-10"]),
                        "valor": [2.5, 0.3, 3.73, 0.2], "tipo": ["dividendo", "jcp", "dividendo", "jcp"],
                        "pagamento": pd.to_datetime(["2025-06-30", "2025-06-30", "2026-12-31", None])})
    por = proventos_por_ano(det, [2025, 2026])
    assert por[2025]["dividendos"] == 2.5 and por[2025]["jcp"] == 0.3 and por[2026]["jcp"] == pytest.approx(0.2)
    fii = pd.DataFrame({"data_com": pd.to_datetime(["2025-01-31", "2025-01-31"]), "valor": [1.1, 0.5],
                        "tipo": ["rendimento", "amortizacao"], "pagamento": pd.to_datetime(["2025-02-14"] * 2)})
    assert proventos_por_ano(fii, [2025])[2025]["rendimentos"] == 1.1
    assert proventos_por_ano(fii, [2025])[2025]["amortizacao"] == 0.5
    prox = proximo_provento(det, "2026-11-23")
    assert prox["valor"] == 3.73 and prox["pagamento"] == pd.Timestamp("2026-12-31")


def test_pares_fii():
    class F:
        def __init__(self, seg, dy, pvp, liq):
            self.segmento, self.dividend_yield, self.pvp, self.liquidez = seg, dy, pvp, liq
    todos = {"HGCR11": F("Títulos e Val. Mob.", .138, .87, 4e6), "KNCR11": F("Títulos e Val. Mob.", .121, 1.01, 9e6),
             "HGLG11": F("Logística", .09, 1.0, 8e6), "ZZZZ11": F("Títulos e Val. Mob.", .1, .9, 1e4)}
    p = pares_fii(todos, "HGCR11.SA")
    assert list(p["papel"]) == ["KNCR11", "HGCR11"]


def test_sinais_dos_indicadores():
    from carteira_analise.raio_x import sinal, sinal_pl, LIMIARES_FII
    assert sinal("roe", 0.247) == "bom" and sinal("roe", 0.05) is None and sinal("roe", -0.02) == "ruim"
    assert sinal("divida_bruta_patrimonio", 1.32) is None and sinal("divida_bruta_patrimonio", 2.0) == "ruim"
    assert sinal("divida_bruta_patrimonio", 0.0) is None  # banco
    assert sinal("liquidez_corrente", 0.8) == "ruim" and sinal("pvp", 1.0) == "bom"
    assert sinal("pvp", 0.9, LIMIARES_FII) == "bom" and sinal("vacancia_media", 0.15, LIMIARES_FII) == "ruim"
    assert sinal("ev_ebitda", 5.6) is None and sinal("roe", None) is None
    assert sinal_pl(8.5, "em linha com o segmento (8,4)") is None
    assert sinal_pl(6.1, "abaixo do segmento (7,6)") == "bom" and sinal_pl(45.8, "acima do setor") == "ruim"
    assert sinal_pl(-3.0, None) == "ruim"


def test_datas_minima_maxima_media_e_queda():
    import pandas as pd
    from carteira_analise.contexto import faixa_12m
    from carteira_analise.raio_x import risco_12m
    d = pd.bdate_range("2025-11-03", periods=260)
    valores = [50.0] * 260
    valores[40] = 60.0   # máxima (topo)
    valores[100] = 30.0  # mínima (fundo)
    s = pd.Series(valores, index=d)
    f = faixa_12m(s)
    assert f["data_minima"] == d[100] and f["data_maxima"] == d[40]
    assert abs(f["media"] - s[s.index > s.index[-1] - pd.Timedelta(days=365)].mean()) < 1e-9
    assert abs(f["preco_x_media"] - (50 / f["media"] - 1)) < 1e-12
    r = risco_12m(s)
    assert r["data_topo"] == d[40] and r["data_fundo"] == d[100] and abs(r["maior_queda"] + 0.5) < 1e-9
