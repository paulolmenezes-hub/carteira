import pandas as pd
import pytest

import carteira_analise.fontes.fundamentus as fdm

HTML = """
<table id="resultado">
<thead><tr><th>Última Data Com</th><th>Tipo</th><th>Data de Pagamento</th><th>Valor</th></tr></thead>
<tbody>
<tr><td>30/09/2026</td><td>Rendimento</td><td>15/10/2026</td><td>1,17</td></tr>
<tr><td>31/08/2026</td><td>Rendimento</td><td>15/09/2026</td><td>1,17</td></tr>
<tr><td>31/08/2026</td><td>Rendimento</td><td>15/09/2026</td><td>0,03</td></tr>
<tr><td>31/07/2026</td><td>Amortização</td><td>14/08/2026</td><td>5,00</td></tr>
<tr><td>30/06/2026</td><td>Rendimento</td><td>14/07/2026</td><td>1,10</td></tr>
<tr><td>29/05/2026</td><td>Rendimento</td><td>15/06/2026</td><td>-</td></tr>
<tr><td>data inválida</td><td>Rendimento</td><td>-</td><td>1,00</td></tr>
<tr><td>30/12/2016</td><td>Rendimento</td><td>13/01/2017</td><td>0,78</td></tr>
</tbody></table>
"""


@pytest.fixture(autouse=True)
def _limpa_cache():
    fdm._cache_proventos.clear()


def test_proventos_pagos_por_data_com():
    s = fdm.proventos_fii_de_html(HTML)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2016-12-30", "2026-06-30", "2026-08-31", "2026-09-30"]
    assert s.loc["2026-08-31"] == pytest.approx(1.20)  # dois rendimentos na mesma data-com: somados
    assert "2026-07-31" not in s.index.strftime("%Y-%m-%d")  # amortização não é rendimento
    assert s.loc["2016-12-30"] == pytest.approx(0.78)


def test_layout_inesperado():
    with pytest.raises(ValueError, match="layout inesperado"):
        fdm.proventos_fii_de_html("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")


def test_busca_com_cache_e_falha(monkeypatch):
    chamadas = []
    monkeypatch.setattr(fdm, "_baixar_html_proventos_fii", lambda papel: chamadas.append(papel) or HTML)
    assert len(fdm.buscar_proventos_fii("hglg11.sa")) == 4
    fdm.buscar_proventos_fii("HGLG11")
    assert chamadas == ["HGLG11"]  # segunda chamada vem do cache

    def falha(papel):
        raise OSError("sem rede")
    monkeypatch.setattr(fdm, "_baixar_html_proventos_fii", falha)
    assert fdm.buscar_proventos_fii("KNRI11") is None


def test_download_monta_url_com_o_papel(monkeypatch):
    urls = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return HTML.encode("latin-1")

    def urlopen(req, timeout=20):
        urls.append(req.full_url)
        return Resp()

    monkeypatch.setattr(fdm.urllib.request, "urlopen", urlopen)
    assert "Rendimento" in fdm._baixar_html_proventos_fii("HGLG11")
    assert urls == ["https://www.fundamentus.com.br/fii_proventos.php?papel=HGLG11&tipo=2"]
