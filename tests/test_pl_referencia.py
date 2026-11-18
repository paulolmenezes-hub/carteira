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
    # 6,1 / 7,94 = 0,77: 15% ou mais abaixo do segmento
    assert ctx["indicadores"][0]["leitura"] == "abaixo do segmento (7,9)"
    assert ctx["indicadores"][0]["nivel"] == "bom"
    assert ctx["indicadores"][1]["leitura"] == "neutro"
    assert ctx["regua_pl"].startswith("Referência: segmento Bancos 7,9 · bolsa 9,4.")


@pytest.mark.parametrize("pl, grupo, esperado, nivel", [
    (8.7, None, "em linha com a bolsa (8,6)", "neutro"),                     # CPFE3
    (14.1, None, "acima da bolsa (8,6)", "atencao"),                         # CXSE3
    (7.9, ("segmento", "Bancos", 7.5), "em linha com o segmento (7,5)", "neutro"),  # ITSA4
    (5.0, ("setor", "Madeira e Papel", 13.0), "abaixo do setor (13,0)", "bom"),
    (122.4, ("segmento", "Minerais Metálicos", 20.1),                       # VALE3: lucro deprimido
     "acima do segmento (20,1) · lucro de 12 meses muito baixo", "atencao"),
    (20.0, ("segmento", "Minerais Metálicos", 20.1),
     "em linha com o segmento (20,1)", "neutro"),
])
def test_rotulo_relativo_a_referencia(pl, grupo, esperado, nivel):
    from carteira_analise.contexto import acrescentar_referencias_pl, ATENCAO, BOM, NEUTRO
    ctx = {"indicadores": [{"nome": "P/L", "valor": f"{pl:.1f}".replace(".", ","), "leitura": "baixo", "nivel": BOM}]}
    g = {"tipo": grupo[0], "nome": grupo[1], "mediana": grupo[2], "n": 9} if grupo else None
    acrescentar_referencias_pl(ctx, {"grupo": g, "bolsa": {"mediana": 8.6, "n": 300}})
    ind = ctx["indicadores"][0]
    assert ind["leitura"] == esperado
    assert ind["nivel"] == {"bom": BOM, "neutro": NEUTRO, "atencao": ATENCAO}[nivel]


def test_sem_referencia_fica_a_regua_fixa():
    from carteira_analise.contexto import acrescentar_referencias_pl
    ctx = {"indicadores": [{"nome": "P/L", "valor": "8,7", "leitura": "baixo", "nivel": "bom"}]}
    acrescentar_referencias_pl(ctx, {"grupo": None, "bolsa": None})
    assert ctx["indicadores"][0]["leitura"] == "baixo" and "regua_pl" not in ctx


def test_motivo_quando_nao_ha_referencia_do_setor():
    from carteira_analise.contexto import texto_referencias_pl
    t = tabela_pl_de_html(TABELA)
    poucas = t[t["papel"].isin(["CPFE3", "EGIE3"])]
    grupos = [({"tipo": "segmento", "id": "1", "nome": "Energia"}, poucas),
              ({"tipo": "setor", "id": "2", "nome": "Utilidade", "erro": "HTTPError: 429"}, None)]
    ref = referencias_pl("CPFE3", grupos, t)
    assert ref["grupo"] is None
    assert "segmento Energia: 1 empresa(s)" in ref["motivo_sem_grupo"] and "429" in ref["motivo_sem_grupo"]
    assert ref["sem_grupo_por_falha"]  # o setor falhou no download
    assert texto_referencias_pl(ref).endswith("(setor indisponível no momento)")


def test_nota_sem_empresas_comparaveis():
    from carteira_analise.contexto import texto_referencias_pl
    t = tabela_pl_de_html(TABELA)
    poucas = t[t["papel"].isin(["CPFE3", "EGIE3"])]
    grupos = [({"tipo": "segmento", "id": "1", "nome": "Energia"}, poucas),
              ({"tipo": "setor", "id": "2", "nome": "Utilidade"}, poucas)]
    ref = referencias_pl("CPFE3", grupos, t)
    assert not ref["sem_grupo_por_falha"]
    assert texto_referencias_pl(ref).endswith("(sem empresas comparáveis no setor)")


def test_pagina_vazia_conta_como_falha_e_nao_vai_para_o_cache(monkeypatch):
    import io
    from carteira_analise.fontes import fundamentus as fm
    respostas = iter([b"", b"<html><table><tr><td>ok</td></tr></table></html>"])

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(fm.urllib.request, "urlopen", lambda req, timeout=20: _Resp(next(respostas)))
    monkeypatch.setattr(fm, "_cache_html", {})
    texto = fm._baixar_html("https://x/teste-vazia", pausa=0, exigir="<table")
    assert "<table" in texto  # a primeira resposta (vazia) foi descartada


def test_pagina_sempre_vazia_levanta_erro_sem_cache(monkeypatch):
    import io
    from carteira_analise.fontes import fundamentus as fm

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(fm.urllib.request, "urlopen", lambda req, timeout=20: _Resp(b"   "))
    monkeypatch.setattr(fm, "_cache_html", {})
    with pytest.raises(ValueError):
        fm._baixar_html("https://x/sempre-vazia", tentativas=2, pausa=0, exigir="<table")
    assert fm._cache_html == {}
