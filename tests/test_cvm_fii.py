import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from carteira_analise.fontes.cvm_fii import (
    baixar_informes,
    comparar_com_referencia,
    ler_zip_informes,
    montar_base_fii,
    serie_rendimentos,
    serie_valor_patrimonial,
    ticker_do_isin,
)


def _zip(ano, col_cnpj="CNPJ_Fundo", dy_percentual=True, versao_dupla=False):
    """ZIP anual no formato da CVM: HGLG11 (VP 160, DY 0,6% a.m., 1 mi de cotas)."""
    datas = pd.date_range(f"{ano}-01-31", periods=12, freq="ME").strftime("%Y-%m-%d")
    cnpj = "11.728.688/0001-47"
    geral = ["{};Data_Referencia;Versao;Nome_Fundo;Codigo_ISIN".format(col_cnpj)]
    comp = ["{};Data_Referencia;Versao;Patrimonio_Liquido;Cotas_Emitidas;Valor_Patrimonial_Cotas;"
            "Percentual_Dividend_Yield_Mes".format(col_cnpj)]
    ap = ["{};Data_Referencia;Versao;Rendimentos_Distribuir".format(col_cnpj)]
    dy = "0,6" if dy_percentual else "0,006"
    for d in datas:
        geral.append(f"{cnpj};{d};1;CSHG LOGÍSTICA FII;BRHGLGCTF004")
        comp.append(f"{cnpj};{d};1;160000000,00;1000000;160,00;{dy}")
        ap.append(f"{cnpj};{d};1;960000,00")
        if versao_dupla:  # reapresentação: a versão 2 corrige o valor patrimonial
            comp.append(f"{cnpj};{d};2;165000000,00;1000000;165,00;{dy}")
        # fundo sem ISIN de cota (ignorado)
        geral.append(f"00.000.000/0001-00;{d};1;FUNDO X;")
        comp.append(f"00.000.000/0001-00;{d};1;1,00;1;1,00;{dy}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"inf_mensal_fii_geral_{ano}.csv", "\n".join(geral).encode("latin-1"))
        z.writestr(f"inf_mensal_fii_complemento_{ano}.csv", "\n".join(comp).encode("latin-1"))
        z.writestr(f"inf_mensal_fii_ativo_passivo_{ano}.csv", "\n".join(ap).encode("latin-1"))
        z.writestr("leia-me.txt", "ignorar")
    return buf.getvalue()


@pytest.mark.parametrize("isin,ticker", [("BRHGLGCTF004", "HGLG11"), ("brknriCTF003", "KNRI11"),
                                         ("", None), ("BR1234CTF000", None), (None, None)])
def test_ticker_do_isin(isin, ticker):
    assert ticker_do_isin(isin) == ticker


def test_le_zip_e_monta_base_com_nomes_de_coluna_diferentes_entre_anos():
    tabs = [ler_zip_informes(_zip(2023)), ler_zip_informes(_zip(2026, col_cnpj="CNPJ_Fundo_Classe"))]
    assert set(tabs[0]) == {"geral", "complemento", "ativo_passivo"}
    base = montar_base_fii(tabs)
    assert set(base["ticker"]) == {"HGLG11"} and len(base) == 24
    linha = base.iloc[0]
    assert linha["vp_cota"] == pytest.approx(160.0) and linha["dy_mes"] == pytest.approx(0.006)
    assert linha["rend_cota_dy"] == pytest.approx(0.96)
    assert linha["rend_cota_passivo"] == pytest.approx(0.96)


def test_dy_em_fracao_nao_e_dividido_de_novo():
    base = montar_base_fii([ler_zip_informes(_zip(2024, dy_percentual=False))])
    assert base["dy_mes"].iloc[0] == pytest.approx(0.006)


def test_reapresentacao_usa_a_ultima_versao():
    base = montar_base_fii([ler_zip_informes(_zip(2024, versao_dupla=True))])
    assert base["vp_cota"].iloc[0] == pytest.approx(165.0) and len(base) == 12


def test_series_por_ticker():
    base = montar_base_fii([ler_zip_informes(_zip(2024))])
    r = serie_rendimentos(base, "HGLG11.SA")
    assert len(r) == 12 and r.iloc[0] == pytest.approx(0.96)
    assert len(serie_rendimentos(base, "HGLG11", forma="passivo")) == 12
    assert serie_valor_patrimonial(base, "hglg11").iloc[-1] == pytest.approx(160.0)
    assert serie_rendimentos(base, "XPTO11").empty


def test_formato_inesperado_e_vazio():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("inf_mensal_fii_geral_2024.csv", "a;b\n1;2")
        z.writestr("inf_mensal_fii_complemento_2024.csv", "a;b\n1;2")
    with pytest.raises(ValueError, match="formato inesperado"):
        montar_base_fii([ler_zip_informes(buf.getvalue())])
    assert montar_base_fii([]).empty and montar_base_fii([{"geral": pd.DataFrame()}]).empty


def test_comparacao_com_os_valores_pagos():
    meses = pd.date_range("2022-01-31", periods=30, freq="ME")
    cvm = pd.Series(0.96, index=meses)
    pagos = pd.Series(1.00, index=meses + pd.Timedelta(days=10), dtype=float)  # pagos no mês seguinte
    pagos.index = pagos.index.tz_localize("America/Sao_Paulo")
    r = comparar_com_referencia(cvm, pagos)
    assert r["meses"] >= 24 and r["erro_mediano_12m"] == pytest.approx(0.04, abs=0.01)
    assert comparar_com_referencia(cvm.iloc[:5], pagos.iloc[:5])["erro_mediano_12m"] is None
    assert comparar_com_referencia(cvm, pd.Series(dtype=float))["meses"] == 0


def test_baixar_informes_ignora_anos_com_falha():
    def falso(url):
        if "2019" in url:
            raise OSError("fora do ar")
        return _zip(int(url.split("_")[-1][:4]))
    tabs = baixar_informes([2019, 2020, 2021], baixar=falso)
    assert len(tabs) == 2 and len(montar_base_fii(tabs)) == 24
