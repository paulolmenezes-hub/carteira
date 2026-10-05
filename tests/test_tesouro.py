import pandas as pd
import pytest

import carteira_analise.fontes.tesouro as tesouro

CSV = """Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;Taxa Venda Manha;PU Compra Manha;PU Venda Manha;PU Base Manha
Tesouro IPCA+;15/05/2035;02/01/2024;5,60;5,72;1.700,10;1.690,00;1.689,00
Tesouro IPCA+ com Juros Semestrais;15/08/2032;02/01/2024;5,40;5,52;4.300,00;4.290,00;4.289,00
Tesouro IPCA+;15/08/2026;02/01/2024;6,10;6,22;3.000,00;2.990,00;2.989,00
Tesouro Prefixado;01/01/2031;02/01/2024;10,80;10,92;600,00;590,00;589,00
Tesouro Renda+ Aposentadoria Extra;15/12/2049;02/01/2024;5,90;6,02;1.000,00;990,00;989,00
Tesouro IPCA+;15/05/2035;03/01/2024;0,00;5,80;0,00;1.680,00;1.679,00
Tesouro IPCA+ com Juros Semestrais;15/08/2032;03/01/2024;;5,50;;4.280,00;4.279,00
"""


def test_juro_real_mediana_dos_titulos_ipca_de_5_a_15_anos():
    s = tesouro.juro_real_tesouro_ipca(CSV)
    # 02/01: só os de 5-15 anos (2032 e 2035), sem prefixado, Renda+ e o de 2026
    assert s.loc["2024-01-02"] == pytest.approx((0.054 + 0.056) / 2)
    # 03/01: taxa de compra zerada/ausente -> usa a de venda
    assert s.loc["2024-01-03"] == pytest.approx((0.058 + 0.055) / 2)


def test_formato_inesperado():
    with pytest.raises(ValueError):
        tesouro.juro_real_tesouro_ipca("a;b\n1;2\n")


def test_download_com_falha(monkeypatch):
    def falha(*a, **k):
        raise OSError("sem rede")
    monkeypatch.setattr(tesouro.urllib.request, "urlopen", falha)
    assert tesouro.baixar_juro_real_tesouro_ipca() is None


def test_download_ok(monkeypatch):
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return CSV.encode("latin-1")
    monkeypatch.setattr(tesouro.urllib.request, "urlopen", lambda *a, **k: Resp())
    assert len(tesouro.baixar_juro_real_tesouro_ipca()) == 2
