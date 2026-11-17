"""P/L comparado ao segmento (ou setor) e à bolsa (17/11/2026)."""
from __future__ import annotations

import pandas as pd
import pytest

from carteira_analise.fontes.fundamentus import grupos_setoriais_de_html, mediana_pl, referencias_pl, tabela_pl_de_html

DETALHES = """<table><tr><td><span class="txt">Setor</span></td>
<td><span class="txt"><a href="resultado.php?setor=32">Energia El&eacute;trica</a></span></td></tr>
<tr><td><span class="txt">Subsetor</span></td>
<td><span class="txt"><a href="resultado.php?segmento=56">Energia El&eacute;trica</a></span></td></tr></table>"""

TABELA = """<table><thead><tr><th>Papel</th><th>Cotação</th><th>P/L</th><th>Liq.2meses</th></tr></thead><tbody>
<tr><td>CPFE3</td><td>44,02</td><td>8,70</td><td>50.000.000,00</td></tr>
<tr><td>EGIE3</td><td>40,00</td><td>12,10</td><td>80.000.000,00</td></tr>
<tr><td>TAEE11</td><td>35,00</td><td>9,30</td><td>60.000.000,00</td></tr>
<tr><td>CMIG4</td><td>11,00</td><td>5,50</td><td>90.000.000,00</td></tr>
<tr><td>XXXX3</td><td>1,00</td><td>-3,20</td><td>2.000.000,00</td></tr>
<tr><td>YYYY3</td><td>2,00</td><td>7,00</td><td>10.000,00</td></tr>
</tbody></table>"""


def test_links_de_setor_e_segmento():
    grupos = grupos_setoriais_de_html(DETALHES)
    assert [g["tipo"] for g in grupos] == ["segmento", "setor"]
    assert grupos[0] == {"tipo": "segmento", "id": "56", "nome": "Energia Elétrica"}


def test_tabela_e_mediana_filtram_prejuizo_e_iliquidez():
    t = tabela_pl_de_html(TABELA)
    assert t.loc[t["papel"] == "EGIE3", "pl"].iloc[0] == pytest.approx(12.1)
    mediana, n = mediana_pl(t, excluir="CPFE3")  # sem prejuízo (XXXX3) e sem a ilíquida (YYYY3)
    assert n == 3 and mediana == pytest.approx(9.3)


def test_segmento_pequeno_cai_para_o_setor():
    t = tabela_pl_de_html(TABELA)
    grupos = [({"tipo": "segmento", "id": "56", "nome": "Energia"}, t),
              ({"tipo": "setor", "id": "32", "nome": "Utilidade Pública"}, t)]
    ref = referencias_pl("CPFE3", grupos, t)
    assert ref["grupo"]["tipo"] == "setor"  # segmento com 3 empresas (< 5)
    assert ref["bolsa"]["n"] == 4 and ref["bolsa"]["mediana"] == pytest.approx((8.7 + 9.3) / 2)


def test_texto_no_indicador_do_contexto():
    from carteira_analise.contexto import acrescentar_referencias_pl
    ctx = {"indicadores": [{"nome": "P/L", "valor": "6,1", "leitura": "baixo", "nivel": "bom"},
                           {"nome": "RSI 14", "valor": "21", "leitura": "neutro", "nivel": "neutro"}]}
    ref = {"grupo": {"tipo": "segmento", "nome": "Bancos", "mediana": 7.94, "n": 12}, "bolsa": {"mediana": 9.4, "n": 250}}
    acrescentar_referencias_pl(ctx, ref)
    assert ctx["indicadores"][0]["leitura"] == "baixo · segmento Bancos: 7,9 · bolsa: 9,4"
    assert ctx["indicadores"][1]["leitura"] == "neutro"
